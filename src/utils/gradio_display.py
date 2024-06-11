import os
import sys
import json
import yaml
from pymongo import MongoClient
import gradio as gr
import pandas as pd
from dotenv import load_dotenv

# Import libraries for working with language models and Google Gemini
import google.generativeai as genai
from utils.timeline_generator import *

# Load environment variables
load_dotenv()
GEMINI_KEY = os.environ.get('GEMINI_KEY')
genai.configure(api_key=GEMINI_KEY)

# Normally where to do this? (in which function?)
with open("../gradio_config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Initialise mongo client.
mongo_client = MongoClient(config["database"]["uri"])


def gradio_generate_timeline(index):
    print("Starting Timeline Generation\n")
    
    train_database, test_database = load_mongodb()
    
    def count_test_length(test_database):
        count = 0
        for doc in test_database:
            count += 1
        return count

    # Select the test article based on the given index
    test_article = test_database[index-1]
    
    df_train = pd.DataFrame(train_database)
    
    # Validate the index
    if index < 0 or index >= count_test_length(test_database):
        return {"error": "Index out of range"}

    # Run this after gradio workflow tested
    timeline, fail_reason = main_hierarchical(test_article, df_train)
    
    
    # def for_testing():
    #     with open('../public/data_upload/final_timeline.json', 'r') as json_file:
    #         timeline = json.load(json_file)
    #     fail_reason = "NIL"
    #     return timeline, fail_reason
    # timeline, fail_reason = for_testing()
    
    
    # Pull database
    db = mongo_client[config["database"]["name"]]
    
    # Get collection from database
    gen_timeline_documents = db[config["database"]["timelines_collection"]]
    
    test_article_id = test_article['st_id']
    test_article_title = test_article['Title']
    # If timeline should not be generated
    if timeline == "to_generate_error" or timeline == "generate_similar_articles_error":
        
        # Timeline instance to return for error message
        timeline_return = {"Article_id": test_article_id, 
                           "Article_Title": test_article_title, 
                           "error": fail_reason}
        
        # Timeline instance to export to MongoDB
        timeline_export = {"Article_id": test_article_id, 
                           "Article_Title": test_article_title, 
                           "Timeline": "null"}
        try:
            # Insert result into collection
            gen_timeline_documents.insert_one(timeline_export)
            print("Data successfully saved to MongoDB")
        except Exception as error:
            print(f"Unable to save timeline to database. Check your connection the database...\nERROR: {error}\n")
            sys.exit()
            
    else:
        # Convert the timeline to JSON
        timeline_json = json.dumps(timeline)
        timeline_return = {"Article_id": test_article_id, 
                           "Article_Title": test_article_title, 
                           "Timeline": timeline_json}
        timeline_export = timeline_return
        
        # Send the timeline data to MongoDB
        try:
            # Insert result into collection
            gen_timeline_documents.insert_one(timeline_export)
            print("Data successfully saved to MongoDB")
        except Exception as error:
            print(f"Unable to save timeline to database. Check your connection the database...\nERROR: {error}\n")
            sys.exit()
    return timeline_return

def display_timeline(timeline_str):
    print("Displaying timeline on Gradio Interface \n")
    if isinstance(timeline_str, str):
        # Used for testing
        timeline_list = json.loads(timeline_str)
    else:
        timeline_list = timeline_str
    display_list= timeline_list[:len(timeline_list)//2]
    html_content = '''
                    <div style='text-align: center;'><strong>
                    First half of events in the timeline:\n
                    </strong></div>
                    <div style='padding: 10px;'>'''
    for event in display_list:
        html_content += f"<h3>{event['Date']}</h3>"
        html_content += f"<p><strong>Event:</strong> {event['Event']}</p>"
        html_content += f"<p><strong>Contextual Annotation:</strong> {event['Contextual_Annotation']}</p>"
        #Display only the first 60 chars of the first article url
        html_content += "<p><strong>Article IDs:</strong> " + event['Article_URL'][0][0][:60] + "</p>"
        html_content += "<hr>"
    html_content += "</div>"
    return html_content

def display_gradio():
    with gr.Blocks(title="Article Timeline Generator", theme='snehilsanyal/scikit-learn') as gradio_timeline:
        gr.Markdown("""
            <h1 style='text-align: center;'>
            Timeline Generator
            </h1>
            <hr>
            <h3>
            Choose an article index to generate a timeline using the test database.
            </h3>
        """)
        
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    input_test_index = gr.Number(label="Test Article Index. Choose an index from 1-7 (Number of test articles)", value=0)
                    hidden_article_id = gr.Textbox(visible=False)

                    with gr.Row():
                        clear_button = gr.Button("Reset index")
                        generate_button = gr.Button("Generate Timeline")
                    shown_article_title = gr.Textbox(label="Title of chosen article")
                    output_timeline = gr.JSON(label="Generated Timeline in JSON format", visible=False)
                    gr.Markdown('''
                                If Error message is not shown past the 7 second mark, a timeline is necessary for the chosen article. 
                                ''')
                    output_error = gr.Textbox(label="Error Message:")  
                    user_download_button = gr.DownloadButton("Download Generated Timeline")
                
                with gr.Column():
                    show_timeline_button = gr.Button("Show Generated Timeline")
                    output_timeline_HTML = gr.HTML()
        
        clear_button.click(lambda: 0, None, input_test_index)
        
        def handle_generate_timeline(index):
                result = gradio_generate_timeline(index)
                article_id = result['Article_id']
                article_title = result['Article_Title']
                if "error" in result:
                    timeline_error = result["error"]
                    return timeline_error, None, article_id, article_title
                else:
                    
                    timeline = result['Timeline']
                    return "NIL, Press Show Generated Timeline to display generated timeline", timeline, article_id, article_title

        generate_button.click(
                handle_generate_timeline,
                inputs=input_test_index,
                outputs=[output_error, output_timeline, hidden_article_id, shown_article_title]
            )
        show_timeline_button.click(
                display_timeline,
                inputs=output_timeline,
                outputs=output_timeline_HTML
            )
        user_download_button.click(
            inputs=output_timeline
        )
    
    # gradio_timeline.launch(inbrowser=True)        
    return gradio_timeline
