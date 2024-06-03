from utils.hierarchical_clustering import *
from utils.data_cleaner import *
from utils.data_loader import *

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
from concurrent.futures import ThreadPoolExecutor, as_completed


# Load environment variables
from dotenv import load_dotenv
load_dotenv()
GEMINI_KEY = os.environ.get('GEMINI_KEY')
genai.configure(api_key=GEMINI_KEY)


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
from concurrent.futures import ThreadPoolExecutor, as_completed


# Load environment variables
from dotenv import load_dotenv
load_dotenv()
GEMINI_KEY = os.environ.get('GEMINI_KEY')
genai.configure(api_key=GEMINI_KEY)


def to_generate_timeline(test_data):
    llm = genai.GenerativeModel('gemini-1.5-flash-latest')
    class Event(BaseModel):
        score: int = Field(description="The need for this article to have a timeline")
        Reason: str = Field(description = "The main reason for your choice why a timeline is needed or why it is not needed")
            

    output_parser = JsonOutputParser(pydantic_object=Event)

        # See the prompt template you created for formatting
    format_instructions = output_parser.get_format_instructions()

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

    # Create the prompt template
    prompt = PromptTemplate(
        input_variables=["text", "title"],
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
        print("Timeline is necessary for this chosen article.\n")
        return True
    else:
        print("A timeline for this article is not required. \n")
        for part in final_response['Reason'].replace(". ", ".").split(". "):
            print(f"{part}\n")
        print("Hence I gave this a required timeline score of " + str(final_response['score']))
        return False

def get_article_dict(input_list, df_train, df_test):
    llm = genai.GenerativeModel("gemini-1.5-flash-latest")

    # Initialize the generative model
    class Event(BaseModel):
        Article_id: list = Field(description="Article ids that are most relevant for the generation of the timeline")
            

    output_parser = JsonOutputParser(pydantic_object=Event)

    # See the prompt template you created for formatting
    format_instructions = output_parser.get_format_instructions()

    template = '''
    Task Description: Given the following test article, and the relevant tags of that article, and the contents of articles similar to it.
    You will only select the articles that are closest in similarity to the test article, \
    for which i will be able to leverage on to build a timeline upon. 
    Return the article ids for the chosen articles. 
    Ensure that the chosen articles are relevant in terms of geographical location, main topic and whether or not they are talking about the same event or topic.
    {text}

    {format_instructions}
    Check and ensure again that the output follows the format instructions above very strictly. 
    '''

    # Create the prompt template
    prompt = PromptTemplate(
        input_variables=["text"],
        partial_variables={"format_instructions": format_instructions},
        template=template,
    )

    final_prompt = prompt.format(text=input_list)
    response = llm.generate_content(
            final_prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
    new_output = re.search(r'\[[^\]]*\]', response.parts[0].text).group(0)
    article_keys =  json.loads(new_output)
    if not article_keys:
        print("No useful similar articles found in database for timeline generation.\n")
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
    print(similar_articles_dict)
    if not similar_articles_dict:
        print("Inadequate articles found to construct a timeline... Exiting execution now\n")
        sys.exit()
    else:
        # Print results 
        print("-"*80 + "\n")
        print(f"Test Article Title: << {similar_articles_dict['Title']}>>\n")
        print("Supporting Article Titles:")
        for idx in similar_articles_dict['indexes']:
            print(f" - {df_train.loc[idx, 'Title']}")
        print("\n" + "-"*80)
        return similar_articles_dict
        
def generate_and_sort_timeline(similar_articles_dict, df_train, df_test):
    llm = genai.GenerativeModel('gemini-1.5-flash-latest' )
    
    class Event(BaseModel):
        Date: str = Field(description="The date of the event in YYYY-MM-DD format")
        Event: str = Field(description="A detailed description of the important event")
        Article: int = Field(description="The article number from which the event was extracted")

    output_parser = JsonOutputParser(pydantic_object=Event)

    # See the prompt template you created for formatting
    format_instructions = output_parser.get_format_instructions()

    template = '''
    Given an article, containing a publication date, title, and content, your task is to construct a detailed timeline of events leading up to the main event described in the article.
    Begin by thoroughly analyzing the title, content, and publication date of the article to understand the main event in the article. 
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
    
    def generate_individual_timeline(date_text_triples):
        s =  f'Article {date_text_triples[0]}: Publication date: {date_text_triples[1]} Article Text: {date_text_triples[2]}'
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
            result = response.parts[0].text
        else:
            retry_response = llm.generate_content(final_prompt,
                                        safety_settings={
                                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
                                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                                            })
            try:
                result = retry_response.parts[0].text
            except ValueError:
                print("ERROR: There were issues with the generation of the timeline. The timeline could not be generated")
                return
        return result
    
    def process_articles(df_train):
        df_retrieve = df_train.loc[similar_articles_dict['indexes']]
        df_retrieve = pd.concat([df_retrieve, df_test], axis=0).iloc[::-1].reset_index(drop=True)

        # Prepare texts and publication dates
        indiv_numbers = list(range(1,len(df_retrieve)+1))
        indiv_text = df_retrieve['combined'].tolist()
        indiv_dates = df_retrieve['Publication_date'].tolist()
        date_text_triples = list(zip(indiv_numbers, indiv_text, indiv_dates))

        dict_of_timelines = {}
        
        with ThreadPoolExecutor(max_workers=len(date_text_triples)) as executor:
            futures = {executor.submit(generate_individual_timeline, date_text_triple): date_text_triple for date_text_triple in date_text_triples}
            i = 0
            for future in as_completed(futures):
                dict_of_timelines[i] = future.result()
                i += 1
        return dict_of_timelines, df_retrieve
    
    timeline_dic, df_retrieve = process_articles(df_train)
    
    print("The first timeline has been generated\n")
    generated_timeline = []
    for idx, line in timeline_dic.items():
        indiv_timeline = clean_output(line)
        if type(indiv_timeline) == list:
            for el in indiv_timeline:
                generated_timeline.append(el)
        else:
            generated_timeline.append(indiv_timeline)
    
    unsorted_timeline = []
    for event in generated_timeline:
        article_index = event["Article"] - 1
        event["Article_id"] = df_retrieve.iloc[article_index].id
    for event in generated_timeline:
        del event["Article"]
        unsorted_timeline.append(event)  
        
    timeline = sorted(unsorted_timeline, key=lambda x:x['Date'])
    finished_timeline = [event for event in timeline if event['Date'].lower()!= 'nan']
    for i in range(len(generated_timeline)):
        date = generated_timeline[i]['Date']
        if date.endswith('-XX-XX') or date.endswith('00-00'):
            generated_timeline[i]['Date'] = date[:4]
        elif date.endswith('-XX') or date.endswith('00'):
            generated_timeline[i]['Date'] = date[:7]
    return finished_timeline

def enhance_timeline(timeline):
    print("\nProceeding to enhance the timeline...\n")
    llm = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')

    class Event(BaseModel):
            Date: str = Field(description="The date of the event in YYYY-MM-DD format")
            Event: str = Field(description="A detailed description of the event")
            Contextual_Annotation: str = Field(description="Contextual anecdotes of the event.")
            Article_id: list = Field(description="The article id(s) from which the event was extracted")

    parser = JsonOutputParser(pydantic_object=Event)

    template = '''
        You are given a timeline of events, your task is to enhance this timeline by improving its clarity and contextual information.
        IF the same event occurs on the exact same date, merge these events to avoid redundancy, and add the article ids to a list. 
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
            
    def generate_enhanced(batch):
        batch_timeline_text = json.dumps(batch)
        final_prompt = prompt.format(text=batch_timeline_text, format_instructions=parser.get_format_instructions())
        response = llm.generate_content(final_prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
        data = extract_json_from_string(response.parts[0].text)
        return data

    def process_articles(timeline):
        results = []
        batches = split_batches(timeline, max_batch_size=30)
        num_batches = len(batches)

        with ThreadPoolExecutor(max_workers=num_batches) as executor:
            print("Processing batches simultaneously now...\n")
            futures = {executor.submit(generate_enhanced, batch): batch for batch in batches}
            for future in as_completed(futures):
                indiv_batch = future.result()
                for event in indiv_batch:
                    results.append(event)
        return results

    full_enhanced = process_articles(timeline)
    sorted_timeline = sorted(full_enhanced, key=lambda x:x['Date'])
    print("Finished enhancing the timeline\n")
    return sorted_timeline

def save_enhanced_timeline(enhanced_timeline, output_path: str):
    """
    Save the enhanced timeline to a JSON file.

    Parameters:
    enhanced_timeline (list): The enhanced timeline data.
    output_path (str): The file path where the JSON will be saved.
    """
    sorted_events = sorted(enhanced_timeline, key=lambda x: x['Date'])
    json_data = json.dumps(sorted_events, indent=4, ensure_ascii=False)

    # Write the JSON string to a file
    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write(json_data)
    print(f"Enhanced timeline saved to '{output_path}'")

def generate_save_timeline(relevant_articles, df_train, df_test, output_path):
    similar_articles = get_article_dict(relevant_articles, df_train, df_test)
    generated_timeline = generate_and_sort_timeline(similar_articles, df_train, df_test)
    final_timeline = enhance_timeline(generated_timeline)
    save_enhanced_timeline(final_timeline, output_path)
    return final_timeline