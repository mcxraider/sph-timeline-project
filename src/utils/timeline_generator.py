import os
import pandas as pd

# Import libraries for working with language models and Google Gemini
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from json import JSONDecodeError
import re
import json
from datetime import datetime


# Load environment variables
load_dotenv()
GEMINI_KEY = os.environ.get('GEMINI_KEY')
genai.configure(api_key=GEMINI_KEY)



def clean_output(output):
    try:
        updated_timeline = json.loads(output)
        return updated_timeline
    except JSONDecodeError:
        #try 1: Ensuring that the string ends with just the open and close lists brackets
        try:
            new_output = re.search(r'\[[^\]]*\]', output).group(0)
        except AttributeError:
            new_output = re.search(r'\{.*?\}', output, re.DOTALL).group(0)  
        updated_timeline = json.loads(new_output)
        return updated_timeline

def to_generate_timeline(similar_articles_dict):
    # Initialize the generative model
    llm = genai.GenerativeModel('gemini-1.0-pro')
    class Event(BaseModel):
        score: int = Field(description="The need for this article to have a timeline")
            
    output_parser = JsonOutputParser(pydantic_object=Event)

    # See the prompt template you created for formatting
    format_instructions = output_parser.get_format_instructions()

    # Define the template
    template = '''
    You are a highly intelligent AI tasked with analyzing articles to determine whether generating a timeline of events leading up to the key event in the article would be beneficial. Consider the following factors to make your decision:

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
    Your answer will be the need for this article to have a timeline with a score 1 - 5, 1 means unnecessary, 5 means necessary.
    {format_instructions}    
    ANSWER:
'''

    # Create the prompt template
    prompt = PromptTemplate(
        input_variables=["text", "title"],
        partial_variables={"format_instructions": format_instructions},
        template=template,
    )     
        # Define the headline
    headline = similar_articles_dict['Title']
    body = similar_articles_dict['Text']

        # Format the prompt
    final_prompt = prompt.format(title=headline, text=body)

        # Generate content using the generative model
    response = llm.generate_content(
            final_prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
    for i in range(len(response.parts[0].text)):
            if response.parts[0].text[i:i+5] == 'score':
                score = int(response.parts[0].text[i+8])
                if score >=3:
                    return True
                else:
                    return False
                

def generate_timeline(similar_articles_dict, df_train, df_test):
    llm = genai.GenerativeModel('gemini-1.0-pro')
    class Event(BaseModel):
        Date: str = Field(description="The date of the event in YYYY-MM-DD format")
        Event: str = Field(description="A detailed description of the event")
        Article: int = Field(description="The article number from which the event was extracted")

    output_parser = JsonOutputParser(pydantic_object=Event)

    # See the prompt template you created for formatting
    format_instructions = output_parser.get_format_instructions()

    template = '''
    Given a series of articles, each containing a publication date, title, and content, your task is to construct a detailed timeline of events leading up to the main event described in the first article.
    Analyze the First Article: Begin by thoroughly analyzing the title, content, and publication date of the first article to understand the main event.
    Use Subsequent Articles: For each following article, examine the title, content, and publication date. Identify events, context, and any time references such as "last week," "last month," or specific dates.

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

    Series of Articles:
    {text}

    {format_instructions}
    Check and ensure again that the output follows the format instructions above very strictly. 
    '''

    prompt = PromptTemplate(
        input_variables=["text"],
        partial_variables={"format_instructions": format_instructions},
        template=template
    )
    df_retrieve = df_train.loc[similar_articles_dict['indexes']]
    # TIDY UP THIS 
    df_retrieve = pd.concat([df_retrieve, df_test.iloc[[key]]], axis=0)
    df_retrieve = df_retrieve.iloc[::-1].reset_index(drop=True)
    indiv_text = list(df_retrieve.combined.values)
    indiv_dates = list(df_retrieve.Publication_date.values)
    all = []
    for i in range(len(indiv_text)):
        s =  f'Article {i+1}: Publication date: {indiv_dates[i]}  {indiv_text[i]}'
        all.append(s)
    sum_of_text = ", ".join(all) 
    
    final_prompt = prompt.format(text=sum_of_text)
    response = llm.generate_content(final_prompt,
                                   safety_settings={
                                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
                                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                                    }
    )
    timelines = {}
    indiv_text = list(df_retrieve.combined.values)
    indiv_dates = list(df_retrieve.Publication_date.values)
    for i in range(len(indiv_text)):
        s =  f'Article {i+1}: Publication date: {indiv_dates[i]}  {indiv_text[i]}'
        final_prompt = prompt.format(text=s)
        response = llm.generate_content(final_prompt,
                                        safety_settings={
                                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
                                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                                            }
        )
    if '[' or '{' not in response.parts[0].text:
        response = llm.generate_content(final_prompt,
                                    safety_settings={
                                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
                                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                                        }
        )
    #key corresponds to the index in the df_test
    try:
        timelines[i] = response.parts[0].text
    except ValueError:
        timelines[i] = "Timeline could not be generated"
        print(f"Error in timeline generation for key {i}")
    return timelines, df_retrieve
    

        
def sort_and_clean(timelines, df_retrieve):
    generated_timeline = []
    for idx, line in timelines.items():
        indiv_timeline = clean_output(line)
        if type(indiv_timeline) == list:
            for el in indiv_timeline:
                generated_timeline.append(el)
        else:
            generated_timeline.append(indiv_timeline)
    for event in generated_timeline:
        # Retrieve article id of respective events in timeline
        article_index = event["Article"] - 1
        event["Article_id"] = df_retrieve.iloc[article_index].id
        del event["Article"]

    sorted_timeline = sorted(generated_timeline, key=lambda x: datetime.strptime(x['Date'], '%Y-%m-%d'))
    return sorted_timeline