# Standard library imports
import os
import json
import logging
from datetime import datetime

# Third-party library imports
import yaml
import torch
import pandas as pd
import requests
from tqdm import trange
from torch import nn
from pymongo import MongoClient
from sentence_transformers import CrossEncoder

# LangChain and Google Gemini imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Initialize logger
logger = logging.getLogger(__name__)

# Read configuration
config_file = '../config.yaml'
with open(config_file, 'r') as fin:
    config = yaml.safe_load(fin)
# end with
GEMINI_KEY   = os.environ['GEMINI_KEY']
MONGO_URI    = os.environ['MONGO_URI']
HF_KEY       = os.environ['HUGGINGFACE_API_KEY']
EMBEDDER_API = os.environ["HF_EMBEDDING_MODEL_URL"]

genai.configure(api_key=GEMINI_KEY)

# Initialise Mongodb client
mongo_client = MongoClient(MONGO_URI)

# Setup default LLM model
default_llm = genai.GenerativeModel('gemini-1.5-flash-latest')

def load_database():
    """
    Connect to the MongoDB client, fetch training and testing data from specified collections,
    combine them into a single DataFrame, and convert it to a dictionary.

    Returns:
        list: A list of dictionaries where each dictionary represents a document from the combined
        training and test data.
    
    Raises:
        Exception: If there is an error fetching data from MongoDB.
    """
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
    df = pd.concat([df_train, df_test], axis=0)
    
    # Convert it to a dictionary for O(1) look up
    db = df.to_dict(orient='records')
    return db
# end def

def get_test_article(db, test_id):
    """
    Retrieve a test article from the database using an article id

    Args:
        db (list): A list of dictionaries where each dictionary represents a document from the combined
        training and test data.
        test_id (str): The ID of the test article to retrieve.

    Returns:
        dict: The test article matching the provided test ID.
    """
    test_article = []
    for article in db:
        if article['st_id'] == test_id:
            test_article.append(article)
            break
        # end if
    # end for
    return test_article[0]
# end def

def clean_llm_output(llm_output):
    """
    Clean the output from a language model by removing specific characters and parsing it as JSON.

    Args:
        llm_output (object): The output from the language model, expected to have a 'parts' attribute
                             which is a list containing objects with a 'text' attribute.

    Returns:
        dict: The parsed JSON result after cleaning the language model output.
    """
    text = llm_output.parts[0].text.replace("```", '').replace('json','')
    result = json.loads(text)
    return result
# end def


def det_generate_timeline(article, score_threshold=3, llm=default_llm, safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    }):
    """
    Evaluate the necessity of generating a timeline for a given article based on specific criteria.

    Args:
        article (dict): A dictionary containing the article information with keys 'Title' and 'Text'.
        score_threshold (int, optional): The threshold score to determine if a timeline is necessary. Default is 3.
        llm (object, optional): The language model to use for generating content. Default is default_llm.
        safety_settings (dict, optional): Safety settings to use for content generation. Default settings block none.

    Returns:
        dict: A dictionary containing:
            - 'det' (bool): Whether a timeline is appropriate for the article.
            - 'score' (int): The score indicating the necessity of the timeline.
            - 'reason' (str or None): The reason for the decision if a timeline is not required.
    """
    # Initialise Pydantic object to force LLM return format
    class Event(BaseModel):
        score: int = Field(description="The need for this article to have a timeline")
        Reason: str = Field(description="The main reason for your choice why a timeline is needed or why it is not needed")
    
    # Initialise JSON output parser
    output_parser = JsonOutputParser(pydantic_object=Event)

    # Define the template
    template = \
    '''
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

    5. Educational Purposes:
       - Would a timeline provide valuable learning or research information?

    Here is the information for the article:
    Title:{title}
    Text: {text}

    Based on the factors above, decide whether generating a timeline of events leading up to the key event in this article would be beneficial. 
    Your answer will include the need for this article to have a timeline with a score 1 - 5, 1 means unnecessary, 5 means necessary. It will also include the main reason for your choice.
    {format_instructions}    
    ANSWER:
    '''
    
    format_instructions = output_parser.get_format_instructions()

    # Create the prompt template
    prompt = PromptTemplate(
        input_variables=["text", "title"],
        partial_variables={"format_instructions": format_instructions},
        template=template,
    )

    # Define the headline
    headline = article["Title"]
    body     = article["Text"]

    # Format the prompt
    final_prompt = prompt.format(title=headline, text=body)

    # Generate content using the generative model
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
    # end if
    else:
        logger.info("A timeline for this article is not required. \n")
        for part in final_response['Reason'].replace(". ", ".").split(". "):
            logger.info(f"{part}\n")
        # end for
        
        logger.info("Hence I gave this a required timeline score of " + str(score))
        reason = "A timeline for this article is not required. \n" \
                    + "\n" + final_response['Reason'] + "\n" + "\nHence this timeline received a necessity score of " \
                    + str(final_response['score']) + "\n"
    # end else
        return {"det": False, "score": score, "reason": reason}
# end def


def generate_timeline_header(article, llm=default_llm):
    """
    Generate a suitable header for a timeline based on the title of the test article.

    Args:
        article (dict): A dictionary containing the article information with a key 'Title'.
        llm (object, optional): The language model to use for generating content. Default is default_llm.

    Returns:
        str: The generated header for the timeline.
    """
    # Define pydantic object to force LLM return output    
    class timeline_header(BaseModel):
        timeline_header: str = Field(description="Suitable header of a timeline for this article")
    
    # Define prompt
    template = \
'''I would like to create a timeline of events based on the title of an article.
Given a list of article titles below, you are tasked with creating an extremely generalised, suitable name for a timeline for this article that will provide a reader contextual information about a timeline of events regarding the article.
The header should be something that can be generalised to other similar articles.
For instance, if a title is "S’pore Red Cross gives $270k worth of relief aid to victims of Hamas-Israel war in Gaza", the header should be "Relief Aid for Gaza Conflict Victims"

Article Title:
{title}

{format_instructions}
Before you return the answer, ensure and double check that you have adhered the answer format instructions strictly.'''
    
    # Initialise JSON output parser
    output_parser = JsonOutputParser(pydantic_object=timeline_header)

    # Create prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["title"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    
    # Format the prompt
    final_prompt = prompt.format(title=article['Title'])
    # Generate response from LLM
    response = llm.generate_content(final_prompt,
                                    safety_settings= {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE})
    
    # Post process LLM output 
    cleaned_output = clean_llm_output(response)
    
    # Get out timeline in a string
    extracted_header = list(cleaned_output.values())[0]
    return extracted_header
# end def


def dense_embedder(payload):
    """
    Initialise a dense embedder model from Hugging Face Inference API and get the embeddings for the given payload.
    Here MUST USE ALL MPNET embed model

    Args:
        payload (dict): The data to be embedded by the model.

    Returns:
        dict: The response from the embedding API containing the embeddings.
    """
    # The embedder API is the embedding model loaded from Hugging Face Inference API.
    response = requests.post(EMBEDDER_API, headers={"Authorization": f"Bearer {HF_KEY}"}, json=payload)
    return response.json()
# end def


def get_text_similarity(timeline_embedding, train_article):
    """
    Generate the cosine similarity between the text embeddings of two articles.

    Args:
        timeline_embedding (list or np.ndarray): The embedding of the timeline text.
        train_article (dict): The training article containing its text embeddings.

    Returns:
        float: The cosine similarity score between the two embeddings.
    """
    # Initialise cosine similarity function
    cos_sim = nn.CosineSimilarity(dim=0)
    
    # Convert embeddings to torch tensors
    text_embedding = torch.tensor(eval(train_article['embeddings']))
    timeline_embedding = torch.tensor(timeline_embedding)
    
    # Compute similarity score
    similarity_score = cos_sim(timeline_embedding, text_embedding)
    return similarity_score
# end def


def get_title_similarity(timeline_embedding, train_article):
    """
    Generate the cosine similarity between the title embeddings of two articles.

    Args:
        timeline_embedding (list or np.ndarray): The embedding of the timeline text.
        train_article (dict): The training article containing its title embeddings.

    Returns:
        float: The cosine similarity score between the two embeddings.
    """
    # Initialise cosine similarity function
    cos_sim = nn.CosineSimilarity(dim=0)
    
    # Convert embeddings to torch tensors
    title_embedding = torch.tensor(eval(train_article['Title_embeddings']))
    timeline_embedding = torch.tensor(timeline_embedding)
    
    # Compute similarity score
    similarity_score = cos_sim(timeline_embedding, title_embedding)
    return similarity_score
# end def


def articles_ranked_by_text(test_article, timeline_header, database):
    """
    Find the top 20 most similar articles based on their text similarity to the test article.

    Args:
        test_article (dict): The test article containing its title and text.
        timeline_header (str): The header for the timeline of the test article.
        database (list): The database of articles to compare against.

    Returns:
        list: A list of the top 20 most similar articles based on text similarity.
    """
    logger.info("Title of test article: " + test_article['Title'] + "\n")
    logger.info("Computing similarities between article texts...")
    
    # Get the embedding of the timeline header
    timeline_heading_embedding = dense_embedder(timeline_header)
    
    article_collection = []
    for i in trange(len(database)):
        article = {}
        article['id'] = database[i]['st_id']
        article['Title'] = database[i]['Title']
        article['Text'] = database[i]['Text']
        article['Date'] = database[i]['Publication_date']
        article['Article_URL'] = database[i]['article_url']
        
        # Calculate cosine similarity score between the timeline header and the article text
        article['cosine_score'] = get_text_similarity(timeline_heading_embedding, database[i])
        
        article_collection.append(article)
    # end for
    
    # Sort by cosine similarity in descending order
    article_collection.sort(key=lambda x: x['cosine_score'], reverse=True)
    
    # Return the top 20 most similar articles
    return article_collection[:20]
# end def


def articles_ranked_by_titles(timeline_header, database):
    """
    Find the top 20 most similar articles based on their title similarity to the timeline header.

    Args:
        timeline_header (str): The header for the timeline of the test article.
        database (list): The database of articles to compare against.

    Returns:
        list: A list of the top 20 most similar articles based on title similarity.
    """
    logger.info("Computing similarities between article titles...")
    
    # Get the embedding of the timeline header
    timeline_heading_embedding = dense_embedder(timeline_header)
    
    article_collection = []
    for i in trange(len(database)):
        article = {}
        article['id'] = database[i]['st_id']
        article['Title'] = database[i]['Title']
        article['Text'] = database[i]['Text']
        article['Date'] = database[i]['Publication_date']
        article['Article_URL'] = database[i]['article_url']
        
        # Calculate cosine similarity score between the timeline header and the article title
        article['cosine_score'] = get_title_similarity(timeline_heading_embedding, database[i])
        
        article_collection.append(article)
    # end for
    
    # Sort by cosine similarity in descending order
    article_collection.sort(key=lambda x: x['cosine_score'], reverse=True)
    
    # Return the top 20 most similar articles
    return article_collection[:20]
# end def


def combine_similar_articles(similar_by_titles, similar_by_text):
    """
    Combine the similar articles retrieved by text and title embeddings, ensuring no duplicates.

    Args:
        similar_by_titles (list): A list of articles similar by title embeddings.
        similar_by_text (list): A list of articles similar by text embeddings.

    Returns:
        list: A list of unique articles combined from both input lists.
    """
    all_articles = []
    # Combine the results by appending one article from each list simultaneously
    for i in range(len(similar_by_text)):
        all_articles.append(similar_by_text[i])
        all_articles.append(similar_by_titles[i])
    # end for
    
    # Initialize a set to track seen titles
    seen_titles = set()

    # List comprehension to remove duplicates based on the title of the article
    unique_articles = []
    for item in all_articles:
        title = item["Title"]
        if title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(item)
        # end if
    # end for
    return unique_articles
# end def


def re_rank_articles(unique_articles, timeline_header, cross_encoder_model="cross-encoder/ms-marco-TinyBERT-L-2-v2"):
    """
    Re-rank articles based on similarity derived from a cross encoder model.

    Args:
        unique_articles (list): A list of unique articles to be re-ranked.
        timeline_header (str): The header for the timeline of the test article.
        cross_encoder_model (str, optional): The cross encoder model to use for re-ranking. Default is "cross-encoder/ms-marco-TinyBERT-L-2-v2".

    Returns:
        list: A list of articles re-ranked based on cross encoder similarity scores.
    """
    cross_encoder = CrossEncoder(
        cross_encoder_model, max_length=512, device="cpu"
    )

    # Format the timeline header and article for cross encoder
    unranked_articles = [(timeline_header, doc['Text']) for doc in unique_articles]
    
    # Predict similarity scores
    similarity_scores = cross_encoder.predict(unranked_articles).tolist()

    # Assign the list of similarity scores to the individual articles 
    for i in range(len(unique_articles)):
        # Only assign the score if there is a positive relationship between the timeline header and the article
        # Here take 0 as the threshold value. (The threshold of 0 was derived by manual observation of the results)
        if similarity_scores[i] > 0:
            unique_articles[i]['reranked_score'] = similarity_scores[i]
        # end if
    # end for

    # Filter articles to include only those with a reranked score
    combined_articles = [article for article in unique_articles if 'reranked_score' in article]
    return combined_articles
# end def


def generate_event_and_date(article, llm=default_llm):
    """
    Generate the main event and its date from the content of an article using an LLM. (Ideally Gemini 1.5-pro-flash)

    Args:
        article (dict): A dictionary containing the article information with keys 'Text', 'Title', and 'Date'.
        llm (object, optional): The language model to use for generating content. Default is default_llm.

    Returns:
        tuple: A tuple containing:
            - str: The main event of the article.
            - str: The date when the main event occurred in YYYY-MM-DD format.
    """
    article_text = article['Text']
    article_title = article['Title']
    article_date = article['Date']
        
    class main_event(BaseModel):
        main_event: str = Field(description="Main event of the article")
        event_date: str = Field(description="Date when the main event occurred in YYYY-MM-DD")
        
    template = \
    '''
    You are a news article editor. Analyse the article deeply, and describe the main event of the article below in one short sentence.
    Using this main event and the publication date, identify the date when this main event occurred.
    You should use any time references such as "last week," "last month," or specific dates. 
    If the article does not specify the exact date, save the date in the YYYY-MM-XX or YYYY-XX-XX format.
    Do not provide any explanations for your answer.

    Publication Date:
    {date}
    Article Title:
    {title}
    Article Text:
    {text}

    {format_instructions}
    Before you return the answer, ensure and double check that you have adhered to the answer format instructions strictly.
    '''
    
    # Initialise JSON output parser
    output_parser = JsonOutputParser(pydantic_object=main_event)

    # Define prompt 
    prompt = PromptTemplate(
        template=template,
        input_variables=["date", "title", "text"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    
    # Format prompt
    final_prompt = prompt.format(date=article_date,
                                 title=article_title,
                                 text=article_text)
    
    # Generate LLM response 
    response = llm.generate_content(final_prompt,
                                    safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    })
    
    # Post process LLM output
    cleaned_output = clean_llm_output(response)
    
    generated_main_event = cleaned_output['main_event']
    generated_date = cleaned_output['event_date']
    return generated_main_event, generated_date
# end def

def filter_ranked_articles(combined_articles, top_k=12):
    """
    Filter out and retain only the top `top_k` articles based on their re-ranked score, and generate the main event for each article.

    Args:
        combined_articles (list): A list of articles that have been combined and re-ranked.
        top_k (int, optional): The number of top articles to retain. Default is 12.

    Returns:
        list: A list of the top `top_k` articles with generated main events and dates.
    """
    # Sort articles by re-ranked score in descending order
    sorted_articles = sorted(combined_articles, key=lambda x: x['reranked_score'], reverse=True)
    
    # Filter out top `top_k` articles
    if len(sorted_articles) > top_k:
        sorted_articles = sorted_articles[:top_k]
    # end if
    
    logging.info("Generating main events from each article\n")
    
    # Add generated main event for each article
    # use threadpool here
    for i in trange(len(sorted_articles)):
        article = sorted_articles[i]
        main_event, event_date = generate_event_and_date(article)
        sorted_articles[i]['Event'] = main_event
        sorted_articles[i]['Event_date'] = event_date
        
        # Remove the cosine score for cleanliness 
        sorted_articles[i].pop("cosine_score", None)
    # end for
    
    return sorted_articles
# end def

def generate_timeline(filtered_articles):
    """
    Format the dates into a human-readable format and sort the articles by date to generate a timeline.

    Args:
        filtered_articles (list): A list of articles with generated main events and dates.

    Returns:
        list: A list of articles with formatted dates and URLs, sorted by date.
    """
    def format_timeline_date(date_str, formats=['%Y', '%Y-%m-%d', '%Y-%m']):
        """
        Format a date string into a human-readable format.

        Args:
            date_str (str): The date string to be formatted.
            formats (list, optional): A list of date formats to try. Default is ['%Y', '%Y-%m-%d', '%Y-%m'].

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
    
    # Sort articles by event date and format the data
    timeline_data = sorted([{"Event": event['Event'], 
                             "Date": event['Event_date'], 
                             "Article_URL": event['Article_URL'], 
                             "Article_title": event['Title']} 
                            
                            for event in filtered_articles],
                            key=lambda x: x['Date'])
    
    for event in timeline_data:
        # Format date into a human-readable format
        event['Date'] = format_timeline_date(event['Date'])
        
        # Generate URL-title pair for easy access in displaying the timeline
        event['Article_URL'] = [{"url": event['Article_URL'], "title": event['Article_title']}]
    # end for
    
    return timeline_data
# end def

def export_hybrid_timeline(test_article, timeline_data, timeline_header):
    """
    Export the generated hybrid timeline to the MongoDB database.

    Args:
        test_article (dict): A dictionary containing the test article information with keys 'st_id' and 'Title'.
        timeline_data (list): A list of dictionaries representing the timeline data.
        timeline_header (str): The header for the timeline.

    Raises:
        Exception: If there is an error saving the timeline to the database.
    """
    logging.info("Fetching database to store the generated timeline.. \n")
    
    # Pull database from MongoDB
    database = mongo_client[config["database"]["name"]]
        
    # Get collection from database that stores the hybrid timeline
    timeline_documents = database[config["database"]["version_2_collection"]]
        
    article_id = test_article['st_id']
    article_title = test_article['Title']

    # Modify timeline header for readability
    display_header = "Timeline: " + timeline_header
    
    timeline_export = {
        "Article_id": article_id, 
        "Article_Title": article_title, 
        "Timeline_header": display_header,
        "Timeline": json.dumps(timeline_data)
    }
                
    # Insert the timeline data to MongoDB
    try:
        timeline_documents.insert_one(timeline_export)
        logging.info(f"Timeline with article id {article_id} successfully saved to MongoDB")
    except Exception as error:
        logging.error(f"Unable to save timeline to database. Check your connection to the database...\nERROR: {error}\n")
# end def

def main(test_id):
    """
    Main function to generate and export a timeline for a given test article.

    Args:
        test_id (str): The ID of the test article to generate a timeline for.
    """
    # Load database 
    db = load_database()

    # Article to generate timeline for
    test_article = get_test_article(db, test_id)    
        
    # Let LLM decide if a timeline is necessary
    timeline_necessary = det_generate_timeline(test_article)
    if timeline_necessary["det"]:
        timeline_header = generate_timeline_header(test_article)
        articles_by_text = articles_ranked_by_text(test_article, timeline_header, db)
        articles_by_titles = articles_ranked_by_titles(timeline_header, db)
        unique_articles = combine_similar_articles(articles_by_titles, articles_by_text)
        re_ranked_articles = re_rank_articles(unique_articles, timeline_header)
        sorted_articles = filter_ranked_articles(re_ranked_articles)
        generated_timeline = generate_timeline(sorted_articles)
        export_hybrid_timeline(test_article, generated_timeline, timeline_header)
    else:
        logging.info(timeline_necessary["reason"])
    # end if
# end def

if __name__ == "__main__":
    main(test_id="st_1155048")