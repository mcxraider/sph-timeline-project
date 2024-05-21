from pydantic import BaseModel, Field
from datetime import datetime
import logging
import json
import os
import re
from json import JSONDecodeError
import time
from tqdm import trange
from typing import List
from dotenv import load_dotenv


# Import libraries for working with language models and Google Gemini
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


# Load environment variables
load_dotenv()
GEMINI_KEY = os.environ.get('GEMINI_KEY')
genai.configure(api_key=GEMINI_KEY)


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

    data = clean_output(response.parts[0].text)
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
    class Event(BaseModel):
        Date: str = Field(description="The date of the event in YYYY-MM-DD format")
        Event: str = Field(description="A detailed description of the event")
        Contextual_Annotation: str = Field(description="Contextual anecdotes of the event.")
        Article: str = Field(description="The article id from which the event was extracted")
        
    template = '''
    You are given a timeline of events, and an article related to this timeline. 
    Your task is to reference the provided article and enhance this timeline by improving its clarity and contextual information.
    Add to the Contextual Annotations by providing annotations for major events to give additional context and improve understanding,\ 
    however, ensure that the contextual annotations added are not repeats of the contents of the event.

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
    parser = JsonOutputParser(pydantic_object=Event)

    final_timeline = []
    unique_ids = []
    for event in sorted_timeline:
        if event['Article'] not in unique_ids:
            unique_ids.append(event['Article'])

    for i in range(len(unique_ids)):
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
        time.sleep(3)

    return final_timeline
