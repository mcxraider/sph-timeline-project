import gradio as gr
import json
import logging

# Initialize logger
logger = logging.getLogger(__name__)


def display_timeline(timeline_str):
    """
    Generates HTML content to display the first half of events from a given timeline.

    Args:
        timeline_str (str or list): The timeline data in JSON string format or as a list of events.

    Returns:
        str: HTML content for displaying the timeline.
    """
    # Log that the timeline is being displayed on the Gradio interface
    logger.info("Displaying timeline on Gradio Interface \n")
    
    # Check if the input is a string, if so, parse it as JSON to get the list of events
    if isinstance(timeline_str, str):
        timeline_list = json.loads(timeline_str)  # Convert JSON string to Python list
    else:
        timeline_list = timeline_str  # Use the list directly if already provided
    
    # Get the first half of the timeline events
    display_list = timeline_list[:len(timeline_list) // 2]
    
    # Start generating HTML content to display events
    html_content = '''
                    <div style='text-align: center;'><strong>
                    First half of events in the timeline:\n
                    </strong></div>
                    <div style='padding: 10px;'>'''
    
    # Loop through the events in the first half and add them to the HTML content
    for event in display_list:
        html_content += f"<h3>{event['Date']}</h3>"  # Display the date of the event
        html_content += f"<p><strong>Event:</strong> {event['Event']}</p>"  # Display the event description
        # Display only the first 60 characters of the first article URL associated with the event
        html_content += "<p><strong>Article URL:</strong> " + event['Article_URL'][0][0][:60] + "</p>"
        html_content += "<hr>"  # Add a horizontal line between events
    
    # Close the HTML content div
    html_content += "</div>"
    
    # Return the generated HTML content
    return html_content

def user_download_button():
    """
    Creates a download button for downloading the generated timeline.

    Returns:
        gr.DownloadButton: A Gradio download button object.
    """
    return gr.DownloadButton(visible=True)

def display_gradio():
    """
    Creates and launches a Gradio interface for generating and displaying article timelines.

    The interface allows the user to select a test article, generate a timeline,
    display the timeline in HTML format, and download the generated timeline.
    """
    # Create a Gradio Blocks interface with a specified title and theme
    with gr.Blocks(title="Article Timeline Generator", theme='snehilsanyal/scikit-learn') as gradio_timeline:
        
        # Add a Markdown element for the header and introductory text
        gr.Markdown("""
            <h1 style='text-align: center;'>
            Timeline Generator
            </h1>
            <hr>
            <h3>
            Choose an article index to generate a timeline using the test database.
            </h3>
        """)
        
        # Create a column layout within the interface
        with gr.Column():
            # Create a row layout within the column
            with gr.Row():
                # Create a nested column for input elements
                with gr.Column():
                    # Input field for selecting the article index (test article index)
                    input_test_index = gr.Number(label="Test Article Index. Choose an index from 1-7 (Number of test articles)", value=0)
                    hidden_article_id = gr.Textbox(visible=False)  # Hidden textbox to store the article ID
                    
                    # Create a row layout for buttons
                    with gr.Row():
                        clear_button = gr.Button("Reset index")  # Button to reset the selected index
                        generate_button = gr.Button("Generate Timeline")  # Button to generate the timeline
                    
                    shown_article_title = gr.Textbox(label="Title of chosen article")  # Textbox to display the chosen article's title
                    output_timeline = gr.JSON(label="Generated Timeline in JSON format", visible=False)  # Hidden JSON output for the generated timeline
                    
                    # Add a Markdown element with an informational message
                    gr.Markdown('''
                                If Error message is not shown past the 7 second mark, a timeline is necessary for the chosen article. 
                                ''')
                    
                    output_error = gr.Textbox(label="Error Message:")  # Textbox to display error messages
                    user_download_button = gr.DownloadButton("Download Generated Timeline")  # Button to download the generated timeline
                
                # Create a nested column for additional buttons and output
                with gr.Column():
                    show_timeline_button = gr.Button("Show Generated Timeline")  # Button to show the generated timeline in HTML format
                    output_timeline_HTML = gr.HTML()  # HTML output to display the timeline

        # Bind the clear button to reset the article index input to 0 when clicked
        clear_button.click(lambda: 0, None, input_test_index)
        
        def handle_generate_timeline(test_id):
            """
            Handles the generation of the timeline for a given test article ID.

            Args:
                test_id (str): The ID of the test article.

            Returns:
                tuple: A tuple containing an error message, the generated timeline, the article ID, and the article title.
            """
            # Find this function in version_2_timeline.py .
            # Call the generate_timeline function to generate the timeline
            result = generate_timeline(test_id)
            article_id = result['Article_id']  # Extract the article ID from the result
            article_title = result['Article_Title']  # Extract the article title from the result
            
            # Check if the result contains an error
            if "error" in result:
                timeline_error = result["error"]  # Extract the error message
                return timeline_error, None, article_id, article_title  # Return the error and other details
            else:
                timeline = result['Timeline']  # Extract the generated timeline
                return "NIL, Press Show Generated Timeline to display generated timeline", timeline, article_id, article_title  # Return the timeline and details

        # Bind the generate button to the handle_generate_timeline function
        generate_button.click(
            handle_generate_timeline,  # Function to call when the button is clicked
            inputs=input_test_index,  # Input element for the function
            outputs=[output_error, output_timeline, hidden_article_id, shown_article_title]  # Output elements for the function
        )

        # Bind the show timeline button to the display_timeline function
        show_timeline_button.click(
            display_timeline,  # Function to call when the button is clicked
            inputs=output_timeline,  # Input element for the function
            outputs=output_timeline_HTML  # Output element for the function
        )

        # Bind the download button to allow downloading of the generated timeline
        user_download_button.click(
            inputs=output_timeline,  # Input element for the function
            outputs=user_download_button  # Output element for the function
        )
    
    # Launch the Gradio interface in the browser
    gradio_timeline.launch(inbrowser=True)


if __name__ == "__main__":
    display_gradio()