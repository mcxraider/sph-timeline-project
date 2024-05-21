# Import necessary libraries
import os
import json
import sys
from typing import Optional
import re
import logging
from json import JSONDecodeError
from typing import Optional
import time
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime


# Import libraries for working with language models and Google Gemini
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Note: Install the google-generativeai package by running the following command in your terminal:
# pip install -U google-generativeai

# Load environment variables
load_dotenv()
GEMINI_KEY = os.environ.get('GEMINI_KEY')
genai.configure(api_key=GEMINI_KEY)
plt.style.use('seaborn')



# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_single_json(file_path: str) -> Optional[dict]:
    """
    Load JSON data from a file.
    
    Parameters:
    file_path (str): The path to the JSON file to be loaded.
    
    Returns:
    Optional[dict]: The loaded JSON data if successful, None otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
        logging.info(f"JSON file '{file_path}' loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error(f"File '{file_path}' not found. Please check the file path.")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file '{file_path}'. Please check the file content.")
        return None

def first_timeline_merge(timeline):
    llm = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')

    class Event(BaseModel):
        Date: str = Field(description="The date of the event in YYYY-MM-DD format")
        Event: str = Field(description="A detailed description of the event")
        Contextual_Annotation: str = Field(description="Contextual anecdotes of the event.")
        Article: str = Field(description="The article id from which the event was extracted")

    parser = JsonOutputParser(pydantic_object=Event)

    template = '''
    You are given a timeline of events, your task is to enhance this timeline by improving its clarity and contextual information.
    If events occur on the same date and have similar descriptions, merge these events to avoid redundancy.
    Add contextual annotations by providing brief annotations for major events to give additional context and improve understanding.
    Only retain important information that would be value-add when the general public reads the information.

    Initial Timeline:
    {text}

    {format_instructions}
    Ensure that the format follows the example output format strictly before returning the output.
    '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template
    )

    final_prompt = prompt.format(text=timeline, format_instructions=parser.get_format_instructions())
    response = llm.generate_content(final_prompt,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
    )

    data = clean_output(response)
    sorted_timeline = sorted(data, key=lambda x: datetime.strptime(x['Date'], '%Y-%m-%d'))
    return sorted_timeline


def clean_output(output):
    try:
        updated_timeline = json.loads(output)
        return updated_timeline
    except JSONDecodeError:
        #try 1: Ensuring that the string ends with just the open and close lists brackets
        output = re.search(r'\[[^\]]*\]', output).group(0)
        updated_timeline = json.loads(output)
        return updated_timeline


def second_timeline_enhancement(sorted_timeline, retrieval):
    llm = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')

    template = '''
    You are given a timeline of events, and an article related to this timeline. 
    Your task is to reference the provided article and enhance this timeline by improving its clarity and contextual information.
    Add to the Contextual Annotations by providing annotations for major events to give additional context and improve understanding, 
    however, ensure that the contextual annotations added do not blatantly repeat what the contents of the event.

    Initial Timeline:
    {timeline}

    Reference Article:
    {article}

    {format_instructions}
    Double check and ensure that the format follows the Example output format strictly before returning the output
    '''
    prompt = PromptTemplate(
        input_variables=["timeline", "article"],
        template=template
    )

    final_timeline = []
    unique_ids = []
    for event in sorted_timeline:
        if event['Article'] not in unique_ids:
            unique_ids.append(event['Article'])

    for i in trange(len(unique_ids)):
        text = [event for event in sorted_timeline if event['Article'] == unique_ids[i]]
        article = retrieval[retrieval['id'] == unique_ids[i]].reset_index()['Text'][0]
        final_prompt = prompt.format(timeline=text, article=article, format_instructions=parser.get_format_instructions())
        response = llm.generate_content(final_prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
        section = clean_output(response.parts[0].text)
        final_timeline.append(section)
        time.sleep(5)

    return final_timeline


def save_enhanced_timeline(enhanced_timeline, output_path: str):
    """
    Save the enhanced timeline to a JSON file.

    Parameters:
    enhanced_timeline (list): The enhanced timeline data.
    output_path (str): The file path where the JSON will be saved.
    """
    unsorted_timeline = []
    for timeline in enhanced_timeline:
        for event in timeline:
            unsorted_timeline.append(event)

    sorted_events = sorted(unsorted_timeline, key=lambda x: x['Date'])
    json_data = json.dumps(sorted_events, indent=4, ensure_ascii=False)

    # Write the JSON string to a file
    with open(output_path, 'w', encoding='utf-8') as fin:
        fin.write(json_data)
    logging.info(f"Enhanced timeline saved to '{output_path}'")



def main():
    single_timeline = '../data_upload/single_timeline_trial.json'
    single_timeline_articles = '../data_upload/df_retrieve.json'
    output_path = '../data_upload/enhanced_timeline_trial.json'

    timeline_data = load_single_json(single_timeline)
    article_data = load_single_json(single_timeline_articles)

    if article_data is not None:
        retrieval = pd.DataFrame(article_data)
    else:
        retrieval = pd.DataFrame()
        logging.warning("No data loaded for retrieval, check file")

    if timeline_data is not None:
        sorted_timeline = first_timeline_merge(timeline_data)
        final_timeline = second_timeline_enhancement(sorted_timeline, retrieval)
        save_enhanced_timeline(final_timeline, output_path)
    else:
        logging.warning("No timeline data to merge.")


if __name__ == "__main__":
    main()
    
