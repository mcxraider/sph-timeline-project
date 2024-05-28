import os
import sys
import re
import json
import pandas as pd

# Import libraries for working with language models and Google Gemini
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from utils.heir_clustering import *
from utils.data_cleaner import *

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
GEMINI_KEY = os.environ.get('GEMINI_KEY')
genai.configure(api_key=GEMINI_KEY)

#DONE
def to_generate_timeline(test_data):
    # Initialize the generative model
    llm = genai.GenerativeModel("gemini-1.5-flash-latest")
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
        input_variables=["title", "text"],
        partial_variables={"format_instructions": format_instructions},
        template=template,
    )     
        # Define the headline
    headline = test_data.Title[0]
    body = test_data.Text[0]

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
    final_response = clean_llm_score(response)
    # If LLM approves
    if final_response['score'] >=3:
        print("Timeline generation is required. Proceeding to generate timeline.")
        return True
    # If LLM doesn't approve 
    else:
        print("A timeline for this article is not required." + "\n" + final_response['Reason'] + "\n" +  "Hence a required timeline score of " + str(final_response['score']))
        return False

#DONE
def get_article_dict(output, df_train, df_test):
    new_output = re.search(r'\[[^\]]*\]', output.text).group(0)
    article_keys =  json.loads(new_output)
    if not article_keys:
        print("No useful similar articles found in database for timeline generation.")
        sys.exit()
    
    similar_articles_dict = {}

    # Iterate over each test article in the filtered df_test
    for index, test_row in df_test.iterrows():
        test_cluster_label = test_row['Cluster_labels']
        
        # Filter df_train for the same cluster label
        df_train_cluster = df_train[df_train['Cluster_labels'] == test_cluster_label]
        
        # Find similar articles in df_train
        similar_indexes = []
        for train_index, train_row in df_train_cluster.iterrows():
            if train_row['id'] in article_keys:
                similar_indexes.append(train_index)
        
        # Store the result in the dictionary if there are at least 2 supporting articles
        if len(similar_indexes) >= 2:
            similar_articles_dict = {
                'Title': test_row['Title'],
                'indexes': similar_indexes,
                'Text': test_row['Text']
            }
            
    # Print results 
    print("-"*80 + "\n")
    print(f"Test Article Title: << {similar_articles_dict['Title']}>>\n")
    print("Supporting Article Titles:")
    for idx in similar_articles_dict['indexes']:
        print(f" - {df_train.loc[idx, 'Title']}")
    print("\n" + "-"*80)
    
    return similar_articles_dict
  
#DONE
def generate_timeline(similar_articles_dict, df_train, df_test):
    llm = genai.GenerativeModel('gemini-1.5-flash-latest' )
    class Event(BaseModel):
        Date: str = Field(description="The date of the event in YYYY-MM-DD format")
        Event: str = Field(description="A detailed description of the important event")
        Article: int = Field(description="The article number from which the event was extracted")

    output_parser = JsonOutputParser(pydantic_object=Event)

    # See the prompt template you created for formatting
    format_instructions = output_parser.get_format_instructions()

    template = '''
    Given an article, containing a publication date, title, and content, your task is to construct a detailed timeline of events leading up to the main event described in the first article.
    Begin by thoroughly analyzing the title, content, and publication date of the article to understand the main event in the first article. 
    the dates are represented in YYYY-MM-DD format. Identify events, context, and any time references such as "last week," "last month," or specific dates. 
    The article could contain more or one key events. 
    If the article does not provide a publication date or any events leading up to the main event, return NAN in the Date field, and 0 i the Article Field

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

    df_retrieve = df_train.loc[similar_articles_dict['indexes']]
    df_retrieve = pd.concat([df_retrieve, df_test], axis=0).iloc[::-1].reset_index(drop=True)

    # Prepare texts and publication dates
    indiv_text = df_retrieve['combined'].tolist()
    indiv_dates = df_retrieve['Publication_date'].tolist()

    timeline_dic = {}
    
    # Do this simultaneously. One worker for each Article. Wont bust the limit
    for i in range(len(indiv_text)):
        s =  f'Article {i+1}: Publication date: {indiv_dates[i]}  {indiv_text[i]}'
        final_prompt = prompt.format(text=s)
        response = llm.generate_content(final_prompt,
                                        safety_settings={
                                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
                                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                                            })
        # Check if Model returns correct format 
        if '[' in response.parts[0].text or '{' in response.parts[0].text:
            timeline_dic[i] = response.parts[0].text
        else:
            retry_response = llm.generate_content(final_prompt,
                                        safety_settings={
                                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
                                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                                            })
            try:
                timeline_dic[i] = retry_response.parts[0].text
            except ValueError:
                print("ERROR: There were issues with the generation of the timeline. The timeline could not be generated")
                sys.exit()  
    
    finished_timeline = clean_sort_timeline(timeline_dic, df_retrieve)
       
    return finished_timeline