import os
import json
import yaml
from tqdm import trange
import torch
from torch import nn
from sentence_transformers import CrossEncoder
from datetime import datetime
from pymongo import MongoClient
import requests

import logging
logger = logging.getLogger(__name__)

import gradio as gr
import pandas as pd
from dotenv import load_dotenv


# Import libraries for working with language models and Google Gemini
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate

# Read configuration
config_file = '../config.yaml'
with open(config_file, 'r') as fin:
    config = yaml.safe_load(fin)
# end with

load_dotenv()
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
    # Connect to the MongoDB client
    try:
        db = mongo_client[config["database"]["name"]]
        train_documents = db[config["database"]["train_collection"]].find()
        logger.info("Train data successfully fetched from MongoDB\n")
    except Exception as error: 
        logger.error(f"Unable to fetch train data from MongoDB. Check your connection the database...\nERROR: {error}\n")
    
    try:
        test_docs = db[config["database"]["test_collection"]].find()
        logger.info("Test data successfully fetched from MongoDB\n")
    except:
        logger.error(f"Unable to fetch test data from MongoDB. Check your connection the database...\nERROR: {error}\n")
    
    df_train = pd.DataFrame.from_dict(list(train_documents))
    df_test = pd.DataFrame.from_dict(list(test_docs))
    
    # Row bind the training and test dataframes 
    df = pd.concat([df_train, df_test], axis=0)
    
    # Convert it to a dictionary for O(1) look up
    db = df.to_dict(orient='records')
    return db
# end def

def get_test_article(db, test_id):
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
        text = llm_output.parts[0].text.replace("```", '').replace('json','')
        result = json.loads(text)
        return result
# end def

def det_generate_timeline(input_data,score_threshold=3,llm=default_llm,
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    }):
    
    """Evaluating necessity of Timeline for this article."""

    # Initialise Pydantic object to force LLM return format
    class Event(BaseModel):
        score: int = Field(description="The need for this article to have a timeline")
        Reason: str = Field(description = "The main reason for your choice why a timeline is needed or why it is not needed")
    
    # Initialise Json output parser
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
    
    # See the prompt template you created for formatting
    format_instructions = output_parser.get_format_instructions()

    # Create the prompt template
    prompt = PromptTemplate(
        input_variables   = ["text", "title"],
        partial_variables = {"format_instructions": format_instructions},
        template=template,
    )

    # Define the headline
    headline = input_data["Title"]
    body     = input_data["Text"]

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
                    + "\n" +final_response['Reason'] + "\n"+ "\nHence this timeline received a necessity score of " \
                    + str(final_response['score'])   + "\n"
    # end else
        return {"det": False, "score": score, "reason": reason}
# end def

def get_article_title(article):
    return article['Title']
# end def

# Generate the header of the timeline of the desired article
def generate_timeline_header(article,llm=default_llm,):
    
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
    
    # Initialise Json output parser
    output_parser = JsonOutputParser(pydantic_object=timeline_header)

    # Create prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["title"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    
    # Format the prompt
    final_prompt = prompt.format(title=get_article_title(article))
    # Generate response from LLM
    response = llm.generate_content(final_prompt,
                                    safety_settings= {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE})
    
    # Post process LLM output 
    cleaned_output = clean_llm_output(response)
    extracted_header= list(cleaned_output.values())[0]
    return extracted_header
# end def

# Initialise dense embedder model from huggingface
def dense_embedder(payload):
    # The embedder API is the model loaded from Hugging face Inference API.
    response = requests.post(EMBEDDER_API, headers={"Authorization": f"Bearer {HF_KEY}"}, json=payload)
    return response.json()
# end def

# Generate the cosine similarity between texts of 2 articles
def get_text_similarity(timeline_embedding,train_article):
    
    cos_sim = nn.CosineSimilarity(dim=0)
    
    text_embedding = torch.tensor(eval(train_article['embeddings']))
    timeline_embedding = torch.tensor(timeline_embedding)
    
    similarity_score = cos_sim(timeline_embedding, text_embedding)
    return similarity_score
# end def

# Generate the cosine similarity between titles of 2 articles
def get_title_similarity(timeline_embedding,train_article):
    
    cos_sim = nn.CosineSimilarity(dim=0)
    
    title_embedding = torch.tensor(eval(train_article['Title_embeddings']))
    timeline_embedding = torch.tensor(timeline_embedding)
    
    similarity_score = cos_sim(timeline_embedding, title_embedding)
    return similarity_score
# end def

# Find the top 20 most similar articles based on their text
def articles_ranked_by_text(test_article,timeline_header,database):
    
    logger.info("Title of test article: " + test_article['Title'] + "\n")
    logger.info("Computing similarities between article texts...")
    
    timeline_heading_embedding = dense_embedder(timeline_header)
    
    article_collection = []
    for i in trange(len(database)):
        article = {}
        article['id'] = database[i]['st_id']
        article['Title'] = database[i]['Title']
        article['Text'] = database[i]['Text']
        article['Date'] = database[i]['Publication_date']
        article['Article_URL'] = database[i]['article_url']
        article['cosine_score'] = get_text_similarity(timeline_heading_embedding, database[i])
        article_collection.append(article)
    # end for
    
    # Sort by cosine similarity in descending order
    article_collection.sort(key = lambda x: x['cosine_score'], reverse=True)
    
    # Returns the top 20 most similar articles
    return article_collection[:20]
# end def

# Find the top 20 most similar articles based on their title
def articles_ranked_by_titles(timeline_header,database):
    
    logger.info("Computing similarities between article titles...")
    
    timeline_heading_embedding = dense_embedder(timeline_header)
    
    article_collection = []
    for i in trange(len(database)):
        article = {}
        article['id'] = database[i]['st_id']
        article['Title'] = database[i]['Title']
        article['Text'] = database[i]['Text']
        article['Date'] = database[i]['Publication_date']
        article['Article_URL'] = database[i]['article_url']
        article['cosine_score'] = get_title_similarity(timeline_heading_embedding, database[i])
        article_collection.append(article)
    
    # Sort by cosine similarity in descending order
    article_collection.sort(key = lambda x: x['cosine_score'], reverse=True)
    
    # Returns the top 20 most similar articles (might not always need top 10)
    return article_collection[:20]
# end def

# Combine the similar articles retrieved by text and title embeddings
def combine_similar_articles(similar_by_titles,similar_by_text):
    
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

# Re rank articles based on similarity derived from cross encoder
def re_rank_articles(unique_articles,timeline_header,cross_encoder_model="cross-encoder/ms-marco-TinyBERT-L-2-v2"):
 
    cross_encoder = CrossEncoder(
        cross_encoder_model, max_length=512, device="cpu"
    )

    # Format the timeline header and article for cross encoder
    unranked_articles = [(timeline_header, doc['Text']) for doc in unique_articles]
    
    # Predict similarity_scores
    similarity_scores = cross_encoder.predict(unranked_articles).tolist()

    # Assign the list of similarity scores to the inidividual articles 
    for i in range(len(unique_articles)):
        # Only assign the score if there is a positive relationship between the timeline header and the article
        if similarity_scores[i]>0:
            unique_articles[i]['reranked_score'] = similarity_scores[i]
        # end if
    # end for
    combined_articles = [article for article in unique_articles if 'reranked_score' in article]
    return combined_articles
# end def

# Generate the main event of each article by using its contents
def generate_event_and_date(article,llm=default_llm,):
    
    article_text = article['Text']
    article_title = article['Title']
    article_date = article['Date']
        
    class main_event(BaseModel):
        main_event: str = Field(description="Main event of the article")
        event_date: str = Field(description="Date which the main event occured in YYYY-MM-DD")
        
    template = \
    '''
    You are a news article editor. Analyse the article deeply, and describe the main event of the article below in one short sentence.
    Using this main event and the publication date, identify the date at when this main event occured.
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
    Before you return the answer, ensure and double check that you have adhered the answer format instructions strictly.
    '''
    
    # Initialise Json output parser
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

# Filter out only the top 12 articles 
def filter_ranked_articles(combined_articles,top_k = 12):
    sorted_articles = sorted(combined_articles, key=lambda x: x['reranked_score'], reverse=True)
    
    # Filter out top 12 articles 
    if len(sorted_articles) > top_k:
        sorted_articles = sorted_articles[:top_k]
    # end if
    
    logging.info("Generating main events from each article\n")
    
    # Add generated main event for each article
    for i in trange(len(sorted_articles)):
        article = sorted_articles[i]
        main_event, event_date = generate_event_and_date(article)
        sorted_articles[i]['Event'] = main_event
        sorted_articles[i]['Event_date'] = event_date
        
        # Remove the cosine score for cleanliness 
        sorted_articles[i].pop("cosine_score")
    # end for
    
    return sorted_articles
# end def

# Format the dates into human readable format and sort by date 
def generate_timeline(filtered_articles):
    
    def format_timeline_date(date_str, formats=['%Y', '%Y-%m-%d', '%Y-%m']):
        """Formats a date string into a human-readable format.
        
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
    
    
    timeline_data = sorted([{"Event": event['Event'], "Date": event['Event_date'], "Article_URL": event['Article_URL'], "Article_title": event['Title']} for event in filtered_articles],
                           key= lambda x: x['Date'])
    
    for event in timeline_data:
        # Format data into human readable format
        event['Date'] = format_timeline_date(event['Date'])
        
        # Generate URL-title pair for easy access in displaying the timeline
        event['Article_URL'] = [{"url": event['Article_URL'], "title": event['Article_title']}]
    # end for
    return timeline_data
# end def

def export_hybrid_timeline(test_article,timeline_data,timeline_header):
    
    logging.info("Fetching database to store the generated timeline.. \n")
    
    # Pull database from MongoDB
    database = mongo_client[config["database"]["name"]]
        
    # Get collection from database that stores the hybrid timeline
    timeline_documents = database[config["database"]["hybrid_timeline_collection"]]
        
    article_id    = test_article['st_id']
    article_title = test_article['Title']

    # Modify timeline header for readability
    display_header = "Timeline: " + timeline_header
    
    timeline_export = {"Article_id": article_id, 
                        "Article_Title": article_title, 
                        "Timeline_header": display_header,
                        "Timeline": json.dumps(timeline_data)}
                
    # Insert the timeline data to MongoDB
    try:
        timeline_documents.insert_one(timeline_export)
        logging.info(f"Timeline with article id {article_id} successfully saved to MongoDB")
    except Exception as error:
        logging.error(f"Unable to save timeline to database. Check your connection the database...\nERROR: {error}\n")
# end def


def main(test_id):
    # Load database 
    db= load_database()

    # Article to generate timeline for
    test_article = get_test_article(db, test_id)    
        
    # Let LLM decide if a timeline is necessary
    timeline_necessary = det_generate_timeline(test_article)
    if timeline_necessary["det"]:
            timeline_header = generate_timeline_header(test_article)
            articles_by_text = articles_ranked_by_text(test_article,timeline_header,db)
            artiles_by_titles = articles_ranked_by_titles(timeline_header,db)
            unique_articles = combine_similar_articles(artiles_by_titles, articles_by_text)
            re_ranked_articles = re_rank_articles(unique_articles, timeline_header)
            sorted_articles = filter_ranked_articles(re_ranked_articles)
            generated_timeline = generate_timeline(sorted_articles)
            export_hybrid_timeline(test_article, generated_timeline, timeline_header)
    else:
            logging.info(timeline_necessary["reason"])


if __name__ == "__main__":
    main(test_id="st_1155048")
    
    