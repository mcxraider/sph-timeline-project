# Standard library imports
import os
import sys
import ast
import json
import re
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError

# Third-party library imports
import yaml
import torch
import numpy as np
import pandas as pd
from torch import nn
from pymongo import MongoClient
from tqdm import trange
from scipy.cluster.hierarchy import linkage, fcluster
from sentence_transformers import CrossEncoder

# LangChain and Google Gemini imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Initialize logger
logger = logging.getLogger(__name__)

# Read configuration
config_file = '../config.yaml'
with open(config_file, 'r') as fin:
    config = yaml.safe_load(fin)
# end with

GROQ_API_KEY = os.environ['GROQ_API_KEY']
GEMINI_KEY   = os.environ['GEMINI_KEY']
MONGO_URI    = os.environ['MONGO_URI']
HF_KEY       = os.environ['HUGGINGFACE_API_KEY']
EMBEDDER_API = os.environ["HF_EMBEDDING_MODEL_URL"]

chat_model = "llama3-8b-8192"
genai.configure(api_key=GEMINI_KEY)

# Initialise Mongodb client
mongo_client = MongoClient(MONGO_URI)

# Setup default LLM model
default_llm = genai.GenerativeModel('gemini-1.5-flash-latest')


def load_database():
    """
    Connects to the MongoDB client, fetches the training and test data collections, 
    and combines them into a single DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the combined training and test data.
    """
    # Connect to the MongoDB client
    try:
        db = mongo_client[config["database"]["name"]]
        train_documents = db[config["database"]["train_collection"]].find()
        logger.info("Train data successfully fetched from MongoDB\n")
    except Exception as error:
        logger.error(f"Unable to fetch train data from MongoDB. Check your connection to the database...\nERROR: {error}\n")

    try:
        test_docs = db[config["database"]["test_collection"]].find()
        logger.info("Test data successfully fetched from MongoDB\n")
    except Exception as error:
        logger.error(f"Unable to fetch test data from MongoDB. Check your connection to the database...\nERROR: {error}\n")

    df_train = pd.DataFrame.from_dict(list(train_documents))
    df_test = pd.DataFrame.from_dict(list(test_docs))

    # Row bind the training and test dataframes
    df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

    return df
# end def

def get_text_embeddings(df):
    """
    Extracts and deserializes the text embeddings from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the text data with serialized embeddings.

    Returns:
        np.ndarray: A NumPy array containing the deserialized embeddings.
    """
    logger.info("Fetching embeddings...\n")
    # Deserializing the embeddings
    body_embeddings = np.array(df['embeddings'].apply(ast.literal_eval).tolist())
    return body_embeddings
# end def

def split_data(test_id):
    """
    Splits the data into training and test sets based on the provided test article ID.

    Args:
        test_id (str or int): The ID of the test article to separate from the training data.

    Returns:
        tuple: A tuple containing the following:
            - pd.DataFrame: The combined DataFrame.
            - pd.DataFrame: The training data DataFrame.
            - pd.DataFrame: The test article DataFrame.
    """
    df = load_database()
    test_article = df[df['st_id'] == test_id].reset_index(drop=True)
    df_train = df[df["st_id"] != test_id]
    return df, df_train, test_article
# end def


def clean_output(output):
    """
    Cleans and parses the JSON output.

    Args:
        output (str): The JSON string to be cleaned and parsed.

    Returns:
        dict or list: The parsed JSON object.

    Raises:
        JSONDecodeError: If the output cannot be parsed as JSON.
    """
    try:
        updated_timeline = json.loads(output)
        return updated_timeline
    except JSONDecodeError:
        # Ensure that the string ends with just the open and close list brackets
        try:
            new_output = re.search(r'\[[^\]]*\]', output).group(0)
        except AttributeError:
            new_output = re.search(r'\{.*?\}', output, re.DOTALL).group(0)  
        updated_timeline = json.loads(new_output)
        return updated_timeline
# end def

def format_timeline_date(date_str, formats=['%Y', '%Y-%m-%d', '%Y-%m']):
    """
    Formats a date string into a human-readable format.

    Args:
        date_str (str): The date string to be formatted.
        formats (list): A list of date formats to try.

    Returns:
        str: The formatted date string, or the original string if no format matches.
    """
    for fmt in formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            if fmt == '%Y':
                return date_obj.strftime('%Y')
            elif fmt == '%Y-%m-%d':
                return date_obj.strftime('%d %B %Y')
            elif fmt == '%Y-%m':
                return date_obj.strftime('%B %Y')
            # end if
        except ValueError:
            continue  # If the format doesn't match, try the next one
    # end for
    
    # If no format matches, return the original date string
    return date_str
# end def

def det_generate_timeline(input_data, score_threshold=3, llm=default_llm,safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    }):
    """
    Evaluates the necessity of generating a timeline for an article.

    Args:
        input_data (dict): The input data containing the article title and text.
        score_threshold (int, optional): The threshold score to decide if a timeline is needed. Defaults to 3.
        llm (object, optional): The language model to use for content generation. Defaults to default_llm.
        safety_settings (dict, optional): The safety settings for the language model. Defaults to blocking none for all categories.

    Returns:
        dict: A dictionary containing the decision on whether a timeline is needed, the score, and the reason if not needed.
    """
    def clean_llm_output(llm_output):
        """
        Cleans the output from the language model.

        Args:
            llm_output (object): The output from the language model.

        Returns:
            dict: The cleaned and parsed JSON output.
        """
        text = llm_output.parts[0].text.replace("```", '').replace('json', '')
        result = json.loads(text)
        return result
    
    # Initialise Pydantic object to enforce LLM return format
    class Event(BaseModel):
        score: int = Field(description="The need for this article to have a timeline")
        Reason: str = Field(description="The main reason for your choice why a timeline is needed or why it is not needed")
    
    # Initialise JSON output parser
    output_parser = JsonOutputParser(pydantic_object=Event)

    # Define the template
    template = '''
    You are a highly intelligent AI tasked with analyzing articles to determine whether generating a timeline of events leading up to the key event in the article would be beneficial. 
    Consider the following factors to make your decision:
    1. **Significance of the Event**:
       - Does the event have a significant impact on a large number of people, industries, or countries?
       - Are the potential long-term consequences of the event important?

    2. **Controversy or Debate**:
       - Is the event highly controversial or has it sparked significant debate?
       - Has the event garnered significant media attention and public interest?

    3. **Complexity**:
       - Does the event involve multiple factors, stakeholders, or causes that make it complex?
       - Does the event have deep historical roots or is it the culmination of long-term developments?

    4. **Personal Relevance**:
       - Does the event directly affect the reader or their community?
       - Is the event of particular interest to the reader due to economic implications, political affiliations, or social issues?

    5. **Educational Purposes**:
       - Would a timeline provide valuable learning or research information?

    Here is the information for the article:
    Title:{title}
    Text: {text}

    Based on the factors above, decide whether generating a timeline of events leading up to the key event in this article would be beneficial. 
    Your answer will include the need for this article to have a timeline with a score 1 - 5, 1 means unnecessary, 5 means necessary. It will also include the main reason for your choice.
    {format_instructions}    
    ANSWER:
    '''
    
    # Get the prompt template format instructions
    format_instructions = output_parser.get_format_instructions()

    # Create the prompt template
    prompt = PromptTemplate(
        input_variables=["text", "title"],
        partial_variables={"format_instructions": format_instructions},
        template=template,
    )

    # Define the headline and body from input data
    headline = input_data["Title"]
    body = input_data["Text"]

    # Format the prompt
    final_prompt = prompt.format(title=headline, text=body)

    # Generate content using the language model
    response = llm.generate_content(
        final_prompt,
        safety_settings=safety_settings,
    )
    final_response = clean_llm_output(response)

    score = final_response['score']
    
    # If LLM approves
    if score >= score_threshold:
        logger.info("Timeline is appropriate for this chosen article.\n")
        return {"det": True, "score": final_response['score'], "reason": None}
    else:
        logger.info("A timeline for this article is not required. \n")
        for part in final_response['Reason'].replace(". ", ".").split(". "):
            logger.info(f"{part}\n")
        # end for
        
        logger.info("Hence I gave this a required timeline score of " + str(score))
        reason = (
            "A timeline for this article is not required. \n"
            + "\n" + final_response['Reason'] + "\n"
            + "\nHence this timeline received a necessity score of "
            + str(final_response['score']) + "\n"
        )
        return {"det": False, "score": score, "reason": reason}
# end def

def get_text_similarity(test_embedding, db_embedding):
    """
    Computes the cosine similarity between the embeddings of a test article and a database article.

    Args:
        test_embedding (list or np.ndarray): The embedding of the test article.
        db_embedding (list or np.ndarray): The embedding of the database article.

    Returns:
        torch.Tensor: The cosine similarity score between the two embeddings.
    """
    cos_sim = nn.CosineSimilarity(dim=0)
    
    # Convert embeddings to tensors
    similarity_score = cos_sim(torch.tensor(test_embedding), torch.tensor(db_embedding))
    return similarity_score
# end def

def articles_ranked_by_text(test_article, cluster_database):
    """
    In the predicted cluster, finds the top 20 most similar articles based on their text embeddings.

    Args:
        test_article (pd.DataFrame): The test article containing the 'embeddings' column.
        cluster_database (pd.DataFrame): The cluster_database of articles.

    Returns:
        list: A list of the top 20 most similar articles.
    """
    logger.info("Computing similarities between article texts...")
    test_embedding = test_article['embeddings'].apply(ast.literal_eval)[0]

    article_collection = []
    for i in trange(len(cluster_database)):
        article = {
            'id': cluster_database.iloc[i]['st_id'],
            'Title': cluster_database.iloc[i]['Title'],
            'Text': cluster_database.iloc[i]['Text'],
            'Date': cluster_database.iloc[i]['Publication_date'],
            'Article_URL': cluster_database.iloc[i]['article_url']
        }
        article_embedding = pd.DataFrame(cluster_database.iloc[i]).loc['embeddings'].apply(ast.literal_eval)[i]
        article['cosine_score'] = get_text_similarity(test_embedding, article_embedding)
        article_collection.append(article)
    # end for
    
    # Sort by cosine similarity in descending order
    article_collection.sort(key=lambda x: x['cosine_score'], reverse=True)
    
    # Return the top 21 most similar articles
    collection = article_collection[:21]
    
    # Remove the first article as it is the test article to get the top 20 articles
    collection.pop(0)
    return collection
# end def

def re_rank_articles(unique_articles, test_article, cross_encoder_model="cross-encoder/ms-marco-TinyBERT-L-2-v2", top_k=6):
    """
    Re-ranks articles based on similarity derived from a cross encoder.

    Args:
        unique_articles (list): The list of unique articles to be re-ranked.
        test_article (pd.DataFrame): The test article.
        cross_encoder_model (str, optional): The cross encoder model to use. Defaults to "cross-encoder/ms-marco-TinyBERT-L-2-v2".
        top_k (int, optional): The number of top articles to return. Defaults to 6.

    Returns:
        list: A list of the top re-ranked articles.
    """
    cross_encoder = CrossEncoder(cross_encoder_model, max_length=512, device="cpu")
    test_article_text = test_article['Text'][0]
    
    # Format the timeline header and article for cross encoder
    unranked_articles = [(test_article_text, doc['Text']) for doc in unique_articles]
    
    # Predict similarity scores
    similarity_scores = cross_encoder.predict(unranked_articles).tolist()
    
    # Assign the list of similarity scores to the individual articles
    for i in range(len(unique_articles)):
        unique_articles[i]['reranked_score'] = similarity_scores[i]
    # end for
    
    combined_articles = [article for article in unique_articles if 'reranked_score' in article]
    best_articles = sorted(combined_articles, key=lambda x: x['reranked_score'], reverse=True)[:top_k]

    return best_articles
# end def

def get_article_dict(test_article, best_articles, df):
    """
    Generates the clustering properties of the test article.

    Args:
        test_article (pd.DataFrame): The test article DataFrame.
        best_articles (list): A list of the best articles.
        df (pd.DataFrame): The DataFrame containing all articles.

    Returns:
        dict: A dictionary containing the clustering properties of the test article.
              If there are insufficient relevant articles, returns "generate_similar_error".
    """
    # Get the DataFrame IDs of the best articles
    ids = [article['id'] for article in best_articles]
    
    similar_indexes = []
    for idx, row in df.iterrows():
        if row['st_id'] in ids:
            similar_indexes.append(idx)
        # end if
        if len(similar_indexes) == len(ids):
            break
        # end if
    # end for

    similar_articles_dict = {
        'Title': test_article['Title'][0],
        'indexes': similar_indexes,
        'Text': test_article['Text'][0],
    }
    
    if len(similar_articles_dict) < 2:
        logger.info("There are insufficient relevant articles to construct a meaningful timeline. ... Exiting execution now\n")
        return "generate_similar_error"
    # end if
    
    return similar_articles_dict
# end def

def get_predicted_cluster(dataframe, test_id):
    """
    Generates the predicted cluster label for the test article.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing all articles.
        test_id (str or int): The ID of the test article.

    Returns:
        int or str: The predicted cluster label for the test article.
    """
    test_article = dataframe[dataframe['st_id'] == test_id].reset_index(drop=True)
    predicted_cluster = test_article['Cluster_label'][0]
    return predicted_cluster
# end def

def find_similar_articles(computed_cluster, df, test_article, max_d, test_id):
    """
    Performs hierarchical clustering and re-ranking to find similar articles.

    Args:
        pre_computed_cluster (np.ndarray): The pre-computed cluster linkage matrix.
        df (pd.DataFrame): The DataFrame containing all articles.
        test_article (pd.DataFrame): The test article DataFrame.
        max_d (float): The maximum distance threshold for clustering.
        test_id (str or int): The ID of the test article.

    Returns:
        dict: A dictionary containing the similar articles and their clustering properties.
    """
    # fcluster returns a np array of the labels
    cluster_labels = fcluster(computed_cluster, max_d, criterion='distance') # SHIFT THIS LINE OUT OF THIS FUNCTION
    df['Cluster_label'] = cluster_labels
    
    # Get the predicted cluster for the test article
    predicted_cluster = get_predicted_cluster(df, test_id)
    
    # Get members of predicted cluster
    cluster_df = df[df['Cluster_label'] == predicted_cluster].reset_index(drop=True)
    
    # Get similar articles based on text embeddings
    similar_articles_by_text_embedding = articles_ranked_by_text(test_article, cluster_df)
    
    # Re-rank the top 6 similar articles
    best_articles = re_rank_articles(similar_articles_by_text_embedding, test_article)
    
    # Get the clustering properties for the test article
    similar_articles = get_article_dict(test_article, best_articles, df)

    return similar_articles
# end def


def generate_and_sort_timeline(similar_articles_dict, df_train, df_test, llm=default_llm):
    """
    Generates and sorts a timeline of events for an article using a language model.

    Args:
        similar_articles_dict (dict): Dictionary containing similar articles and their indexes.
        df_train (pd.DataFrame): DataFrame containing the training data.
        df_test (pd.DataFrame): DataFrame containing the test data.
        llm (object, optional): The language model to use for content generation. Defaults to default_llm.

    Returns:
        list: A list of sorted timeline events.
        pd.DataFrame: The DataFrame containing the retrieved articles.
    """
    # Define the event model
    class Event(BaseModel):
        Date: str = Field(description="The date of the event in YYYY-MM-DD format")
        Event: str = Field(description="A detailed description of the important event")
        Article: int = Field(description="The article number from which the event was extracted")

    output_parser = JsonOutputParser(pydantic_object=Event)

    # Get the prompt template format instructions
    format_instructions = output_parser.get_format_instructions()

    # Define the template
    template = '''
    Given an article, containing a publication date, title, and content, your task is to construct a detailed timeline of events leading up to the main event described in the article.
    Begin by thoroughly analyzing the title, content, and publication date of the article to understand the main event in the article. 
    The dates are represented in YYYY-MM-DD format. Identify events, context, and any time references such as "last week," "last month," or specific dates. 
    The article could contain more than one key event. 
    If the article does not provide a publication date or any events leading up to the main event, return NAN in the Date field, and 0 in the Article field.

    Construct the Timeline:
    Chronological Order: Organize the events chronologically, using the publication dates and time references within the articles.
    Detailed Descriptions: Provide detailed descriptions of each event, explaining how it relates to the main event of the first article.
    Contextual Links: Use information from the articles to link events together logically and coherently.
    Handle Ambiguities: If an article uses ambiguous time references, infer the date based on the publication date of the article and provide a clear rationale for your inference.

    Contextual Links:
    External Influences: Mention any external influences (e.g., global conflicts, economic trends, scientific discoveries) that might have indirectly affected the events.
    Internal Issues: Highlight any internal issues or developments (e.g., political changes, organizational restructuring, societal movements) within the entities involved that might have impacted the events.
    Efforts for Improvement: Note any indications of efforts to improve the situation (e.g., policy changes, strategic initiatives, collaborative projects) despite existing challenges.

    Be as thorough and precise as possible, ensuring the timeline accurately reflects the sequence and context of events leading to the main event.

    Article:
    {text}

    {format_instructions}
    Check and ensure again that the output follows the format instructions above very strictly.
    '''

    prompt = PromptTemplate(
        input_variables=["text"],
        partial_variables={"format_instructions": format_instructions},
        template=template
    )
    
    def generate_individual_timeline(date_text_triples):
        """
        Generates an individual timeline for an article.

        Args:
            date_text_triples (tuple): A tuple containing article number, publication date, and article text.

        Returns:
            str: The generated timeline.
        """
        article_details = f'Article {date_text_triples[0]}: Publication date: {date_text_triples[1]} Article Text: {date_text_triples[2]}'
        final_prompt = prompt.format(text=article_details)
        response = llm.generate_content(
            final_prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
        
        # Check if Model returns correct format 
        if '[' in response.parts[0].text or '{' in response.parts[0].text:
            result = response.parts[0].text
        else:
            retry_response = llm.generate_content(
                final_prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                }
            )
            try:
                result = retry_response.parts[0].text
            except ValueError:
                logger.error("ERROR: There were issues with the generation of the timeline. The timeline could not be generated")
                return
        return result
    # end def
    
    def process_articles(df_train, similar_articles_dict, df_test):
        """
        Processes the articles to retrieve and concatenate dataframes.

        Args:
            df_train (pd.DataFrame): The training data DataFrame.
            similar_articles_dict (dict): Dictionary containing similar articles and their indexes.
            df_test (pd.DataFrame): The test data DataFrame.

        Returns:
            dict: Dictionary of generated timelines.
            pd.DataFrame: DataFrame of retrieved articles.
        """
        # Retrieve and concatenate dataframes
        df_retrieve = pd.concat([df_train.loc[similar_articles_dict['indexes']], df_test], axis=0).iloc[::-1].reset_index(drop=True)

        # Prepare metadata for LLM
        date_text_triples = list(zip(range(1, len(df_retrieve) + 1), df_retrieve['Publication_date'].tolist(), df_retrieve['combined'].tolist()))

        # Generate timelines using ThreadPoolExecutor
        dict_of_timelines = {}
        try:
            with ThreadPoolExecutor(max_workers=len(date_text_triples)) as executor:
                futures = {executor.submit(generate_individual_timeline, triple): i for i, triple in enumerate(date_text_triples)}
                for future in as_completed(futures):
                    dict_of_timelines[futures[future]] = future.result()
        except IndexError:
            with ThreadPoolExecutor(max_workers=len(date_text_triples)) as executor:
                futures = {executor.submit(generate_individual_timeline, triple): i for i, triple in enumerate(date_text_triples)}
                for future in as_completed(futures):
                    dict_of_timelines[futures[future]] = future.result()

        return dict_of_timelines, df_retrieve
    # end def
    
    # Process the articles to generate timelines
    timeline_dic, df_retrieve = process_articles(df_train, similar_articles_dict, df_test)
    logger.info("The first timeline has been generated\n")
    
    # Clean and sort the generated timelines
    generated_timeline = []
    for _, line in timeline_dic.items():
        indiv_timeline = clean_output(line)
        if isinstance(indiv_timeline, list):
            for el in indiv_timeline:
                generated_timeline.append(el)
        else:
            generated_timeline.append(indiv_timeline)
    
    unsorted_timeline = []
    for event in generated_timeline:
        article_index = event["Article"] - 1
        event["Article_id"] = df_retrieve.iloc[article_index].st_id
    for event in generated_timeline:
        del event["Article"]
        unsorted_timeline.append(event)  
    
    timeline = sorted(unsorted_timeline, key=lambda x: x['Date'])
    finished_timeline = [event for event in timeline if event['Date'].lower() != 'nan']
    for i in range(len(finished_timeline)):
        date = finished_timeline[i]['Date']
        # if LLM only knows the year
        if date.endswith('-XX-XX') or date.endswith('00-00'):
            finished_timeline[i]['Date'] = date[:4]
        # if llm only knows the year and month
        elif date.endswith('-XX') or date.endswith('00'):
            finished_timeline[i]['Date'] = date[:7]
    
    return finished_timeline, df_retrieve
# end def

def reduce_by_date(timeline_list, llm=default_llm):
    """
    Takes in a list of events in one day and returns a list of dicts for events in that day,
    filtering out duplicated or redundant events.

    Args:
        timeline_list (list): List of events for a particular day.
        llm (object, optional): The language model to use for content generation. Defaults to default_llm.

    Returns:
        list: A list of dictionaries with filtered and possibly merged events.
    """
    def extract_content_from_json(string):
        """
        Extracts content from a JSON string.

        Args:
            string (str): The JSON string to extract content from.

        Returns:
            list or None: The extracted JSON data or None if extraction fails.
        """
        # Use a regular expression to find the content within the first and last square brackets
        match = re.search(r'\[.*\]', string, re.DOTALL)
        
        if match:
            json_content = match.group(0)
            try:
                # Load the extracted content into a JSON object
                json_data = json.loads(json_content)
                return json_data
            except json.JSONDecodeError as e:
                logger.error("Failed to decode JSON:", e)
                return None
        else:
            logger.info("No valid JSON content found.")
            return None
        # end if
    # end def
    
    timeline_string = json.dumps(timeline_list)
    
    # Define the event model for structured LLM output
    class Event(BaseModel):
        Event: str = Field(description="A detailed description of the event")
        Article_id: list = Field(description="The article id(s) from which the events were extracted")

    parser = JsonOutputParser(pydantic_object=Event)

    # Define the template for the prompt
    template = '''
    You are a news article editor tasked with simplifying a section of a timeline of events on the same day. 
    Given this snippet a timeline, filter out duplicated events. 
    IF events convey the same overall meaning, I want you to merge these events into one event to avoid redundancy, and add the article ids to a list. 
    However, the events are all different, do not combine them, I want you to return it as it is, however, follow the specified format instructions below. 
    Furthermore, if the event is not considered to be an event worthy of an audience reading the timeline, do not include it.
    Take your time and evaluate the timeline slowly to make your decision.

    Timeline snippet:
    {text}

    {format_instructions}
    Ensure that the format follows the example output format strictly before returning the output.'''
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    final_prompt = prompt.format(text=timeline_string)
    
    response = llm.generate_content(
        final_prompt,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
    )
    
    data = extract_content_from_json(response.parts[0].text)
    
    # Limit the number of article links to 2 links
    link_limit = 2
    
    for event_dic in data:
        article_id_list = event_dic['Article_id']
        if len(article_id_list) > link_limit:
            shortened_ids = article_id_list[:link_limit]
            event_dic['Article_id'] = shortened_ids
        # end if
    # end for
    
    return data
# end def

def first_timeline_enhancement(timeline):
    """
    Takes in a timeline in list format and enhances it by combining events with the same date.

    Args:
        timeline (list): A list of events with their dates and descriptions.

    Returns:
        list: An enhanced timeline with combined and filtered events.
    """
    # Get the unique dates seen in the timeline
    unique_dates = sorted(list(set([event['Date'] for event in timeline])))
    
    # Combine the events that have the same date
    dic = {}
    for date in unique_dates:
        dic[date] = [{'Event': event['Event'], 'Article_id': event['Article_id']} for event in timeline if event['Date'] == date]
    # end for
    
    # Combine the events that have the same date and (only if they have the same event)
    new_timeline = {}
    for date, snippet in dic.items():
        if len(snippet) == 1:
            new_timeline[date] = snippet
        else:
            new_snippet = reduce_by_date(snippet)
            new_timeline[date] = new_snippet
        # end if
    # end for
    
    enhanced_timeline = []
    for date, events in new_timeline.items():
        for event in events:
            new_event = {
                'Date': date,
                'Event': event['Event'],
                'Article_id': event['Article_id'] if isinstance(event['Article_id'], list) else [event['Article_id']]
            }
            enhanced_timeline.append(new_event)
        # end for
    # end for
    
    return enhanced_timeline
# end def

def pair_article_urls(enhanced_timeline, df_retrieve):
    """
    Pairs article URLs and titles with events in the enhanced timeline.

    Args:
        enhanced_timeline (list): List of events with article IDs.
        df_retrieve (pd.DataFrame): DataFrame containing article details.

    Returns:
        list: Enhanced timeline with article URLs and titles.
    """
    def edit_timeline(timeline):
        """
        Formats the dates in the timeline.

        Args:
            timeline (list): List of events with dates.

        Returns:
            list: Timeline with formatted dates.
        """
        for event in timeline:
            new_date = format_timeline_date(event['Date'])
            event['Date'] = new_date
        # end for
        return timeline
    # end def

    edited_timeline = edit_timeline(enhanced_timeline)

    # Add URLs and titles to events in the timeline
    for event in edited_timeline:
        id_list = event['Article_id']
        url_title_pairs = []
        
        for i in range(len(id_list)):
            id = id_list[i]
            url = df_retrieve[df_retrieve['st_id'] == id]['article_url'].values[0]
            title = df_retrieve[df_retrieve['st_id'] == id]['Title'].values[0]
            url_title_pairs.append({'url': url, 'title': title})
        # end for
        
        event['Article_URL'] = url_title_pairs
        event.pop('Article_id')
    # end for
    return edited_timeline
# end def

def articles_needing_summary(timeline):
    """
    Identifies events in the timeline that need to be summarized.

    Args:
        timeline (list): List of events in the timeline.

    Returns:
        dict: Dictionary of events that need to be summarized, indexed by their position in the timeline.
    """
    def get_num_words(event_str):
        """
        Counts the number of words in an event description.

        Args:
            event_str (str): Event description.

        Returns:
            int: Number of words in the event description.
        """
        ls = event_str.split()
        return len(ls)
    # end def

    need_summary_timeline = []
    for i in range(len(timeline)):
        # If the number of words in the event is more than 20
        if get_num_words(timeline[i]['Event']) > 20:
            need_summary_timeline.append((i, timeline[i]))
        # end if
    # end for
    
    # Extract events needing summary
    events = {}
    for i in range(len(need_summary_timeline)):
        events[need_summary_timeline[i][0]] = need_summary_timeline[i][1]
    # end for
    return events
# end def

def generate_event_summary(events_ls):
    """
    Summarizes events in the given list of events.

    Args:
        events_ls (list): List of events to be summarized.

    Returns:
        list: List of summarized events.
    """
    # Define summarized event output
    class summarized_event(BaseModel):
        Event: str = Field(description="Event in a timeline")
        Event_Summary: str = Field(description="Short Summary of event")
    
    parser = JsonOutputParser(pydantic_object=summarized_event)

    chat = ChatGroq(temperature=0, model_name=chat_model)
    
    # Template for the prompt
    template = '''
You are a news article editor.
Given a list of events from a timeline, you are tasked to provide a short summary of these series of events. 
For each event, you should return the event, and the summary.

Series of events:
{text}

{format_instructions}
    '''
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | chat | parser
    
    event_str = json.dumps(events_ls)
    result = chain.invoke({"text": event_str})
    if isinstance(result, list):
        return result
    else:
        cleaned_result = clean_output(result)
    # end if
    return cleaned_result
# end def

def merge_event_summaries(events_ls, llm_answer, timeline, need_summary_timeline):
    """
    Merges summarized events back into the original timeline.

    Args:
        events_ls (list): List of events needing summary.
        llm_answer (list): List of summarized events from the language model.
        timeline (list): The original timeline of events.
        need_summary_timeline (dict): Dictionary of events that need summarization.

    Returns:
        list: The updated timeline with summarized events merged in.
    """
    # Check if the length of the summarized events matches the length of events needing summary
    if len(llm_answer) != len(need_summary_timeline):
        logger.info("Groq had an error where timeline summary output length not equal to input but trying to resolve")
        llm_answer = generate_event_summary(events_ls)
        logger.info(len(llm_answer) == len(events_ls))
    # end if
    
    # Merge the summarized events back into the need_summary_timeline
    i = 0
    for k, v in need_summary_timeline.items():
        need_summary_timeline[k]['Event_Summary'] = llm_answer[i]['Event_Summary']
        i += 1
    # end for
        
    # Update the original timeline with the summarized events
    for i in range(len(timeline)):
        if i in need_summary_timeline:
            timeline[i] = need_summary_timeline[i]
        # end if
    # end for
    
    return timeline
# end def

def second_timeline_enhancement(timeline):
    """
    Enhances the timeline by summarizing lengthy events.

    Args:
        timeline (list): The original timeline of events.

    Returns:
        list: The enhanced timeline with summarized events.
    """
    # Identify events that need summarization
    need_summary_timeline = articles_needing_summary(timeline)
    events_ls = [event['Event'] for _, event in need_summary_timeline.items()]
    
    # Generate summaries for the identified events
    summaries = generate_event_summary(events_ls)
    
    # Merge the summaries back into the timeline
    final_timeline = merge_event_summaries(events_ls, summaries, timeline, need_summary_timeline)
    
    return final_timeline
# end def

def generate_save_timeline(similar_articles, df_train, df_test):
    """
    Generates and saves the enhanced timeline.

    Args:
        similar_articles (dict): Dictionary containing similar articles and their indexes.
        df_train (pd.DataFrame): DataFrame containing the training data.
        df_test (pd.DataFrame): DataFrame containing the test data.

    Returns:
        list or str: The final enhanced timeline or an error message.
    """
    if similar_articles == "generate_similar_error":
        return "Error02"
    
    # Generate and sort the initial timeline
    generated_timeline, df_retrieve = generate_and_sort_timeline(similar_articles, df_train, df_test)
    logger.info("Proceeding to Stage 1/2 of enhancement...\n")
    
    # Perform the first enhancement on the timeline
    first_enhanced_timeline = first_timeline_enhancement(generated_timeline)
    
    # Pair article URLs with the enhanced timeline
    second_enhanced_timeline = pair_article_urls(first_enhanced_timeline, df_retrieve)
    logger.info("Proceeding to Stage 2/2 of enhancement...\n")
    
    # Perform the second enhancement on the timeline
    final_timeline = second_timeline_enhancement(second_enhanced_timeline)
    logger.info("Timeline enhanced..\n")
    
    return final_timeline
# end def

def generate_timeline_header(final_timeline):
    """
    Generates a header for the timeline based on the events.

    Args:
        final_timeline (list): The final enhanced timeline of events.

    Returns:
        str: The generated header for the timeline.
    """
    # Extract event titles for headline generation
    event_titles = [event['Event_Summary'] if 'Event_Summary' in event else event['Event'] for event in final_timeline ]
    
    # Define timeline heading pydantic object
    class timeline_heading(BaseModel):
        timeline_heading: str = Field(description="Heading of timeline")
    
    parser = JsonOutputParser(pydantic_object=timeline_heading)

    chat = ChatGroq(temperature=0, model_name=chat_model)
    
    # Template for the prompt
    template = '''
You are a news article editor.
Given a list of events from a timeline, analyze the series of events which are displayed in chronological order. 
Form a concise and accurate heading for this timeline.

Series of events:
{text}

{format_instructions}
    '''
    prompt = PromptTemplate(
        template=template,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | chat | parser
    
    # Invoke the language model to generate the timeline header
    result = chain.invoke({"text": event_titles})
    
    return result
# end def

def main_hierarchical(df, test_article, df_train, test_id):
    """
    Performs hierarchical clustering and generates a timeline if necessary.

    Args:
        df (pd.DataFrame): DataFrame containing all articles.
        test_article (pd.DataFrame): DataFrame containing the test article.
        df_train (pd.DataFrame): DataFrame containing the training data.
        test_id (str): The ID of the test article.

    Returns:
        tuple: A tuple containing the final timeline and a failure reason if applicable.
    """
    embeddings = get_text_embeddings(df)
    max_d = 0.58
    
    # hierarchical clustering
    Z = linkage(embeddings, method='average', metric='cosine')   
    
    # Check if a timeline is necessary for the test article
    timeline_necessary = det_generate_timeline(test_article)
    if timeline_necessary["det"]:
        # Find similar articles using clustering and reranking
        similar_articles = find_similar_articles(Z, df, test_article, max_d, test_id)
        # Generate the final timeline
        final_timeline = generate_save_timeline(similar_articles, df_train, test_article)
        
        if final_timeline == "Error02":
            reason02 = "There are insufficient relevant articles to construct a meaningful timeline."
            return "generate_similar_articles_error", reason02
        # end if
        return final_timeline, None
    else:
        return "to_generate_error", timeline_necessary['reason']
    # end if
# end def

def generate_timeline(test_id):
    """
    Generates a timeline for the specified test article and saves it to MongoDB.

    Args:
        test_id (str): The ID of the test article.

    Returns:
        dict: A dictionary containing the timeline details or an error message.
    """
    logger.info("Starting Timeline Generation\n")
    
    df, df_train, test_article = split_data(test_id)
    
    # Generate the timeline using hierarchical clustering
    timeline, fail_reason = main_hierarchical(df, test_article, df_train, test_id)
    
    logger.info("Fetching database to store the generated timeline...\n")
    # Connect to MongoDB
    db = mongo_client[config["database"]["name"]]
    
    # Get the timelines collection from the database
    gen_timeline_documents = db[config["database"]["version_1_collection"]]
    
    test_article_id = test_article['st_id'][0]
    test_article_title = test_article['Title'][0]
    
    # Handle cases where the timeline should not be generated
    if timeline == "to_generate_error" or timeline == "generate_similar_articles_error":
        # Timeline instance to return for error message
        timeline_return = {
            "Article_id": test_article_id,
            "Article_Title": test_article_title,
            "Timeline_header": "null",
            "error": fail_reason
        }
        
        # Timeline instance to export to MongoDB
        timeline_export = {
            "Article_id": test_article_id,
            "Article_Title": test_article_title,
            "Timeline_header": "null",
            "Timeline": "null"
        }
        try:
            # Insert result into collection
            gen_timeline_documents.insert_one(timeline_export)
            logger.info(f"Timeline with article id {test_article_id} successfully saved to MongoDB")
        except Exception as error:
            logger.error(f"Unable to save timeline to database. Check your connection to the database...\nERROR: {error}\n")
            sys.exit()
    else:
        # Generate a header for the timeline if no error occurred
        logger.info("Generating the timeline header...\n")
        timeline_header = generate_timeline_header(timeline)['timeline_heading']
        # Convert the timeline to JSON
        timeline_json = json.dumps(timeline)
        timeline_return = {
            "Article_id": test_article_id,
            "Article_Title": test_article_title,
            "Timeline_header": timeline_header,
            "Timeline": timeline_json
        }
        timeline_export = timeline_return
        
        # Insert the timeline data into MongoDB
        try:
            gen_timeline_documents.insert_one(timeline_export)
            logger.info(f"Timeline with article id {test_article_id} successfully saved to MongoDB")
        except Exception as error:
            logger.error(f"Unable to save timeline to database. Check your connection to the database...\nERROR: {error}\n")
            sys.exit()
    # end if
    
    return timeline_return
# end def

def main(test_id): 
    """
    Main function to generate a timeline for a specific test article.

    Returns:
        dict: The generated timeline details.
    """
    timeline_return = generate_timeline(test_id)
    return timeline_return
# end def

if __name__=="__main__":
    timeline = main(test_id="st_1155048")