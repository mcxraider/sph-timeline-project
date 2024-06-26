# PURPOSE OF SCRIPT: To extract summaries of articles from 2023-08 to 2023-11 (Link: https://drive.google.com/drive/folders/1BIorb240iHyTK3gXqG5QHAhlKC0FOZbX)
#   - This script is first ran, then troubleshooter.ipynb is created to re-prompt the local llama3 model for erroneous outputs from this script
#   - Examples of erroneous output: error parsing model outputs for certain rows and null .json files
#   - Running this script should generate many .json files, each .json file will be a summary of an article
#   - Compiled them into 1 .json file in troubleshooter.ipynb
#   - BEFORE RUNNING SCRIPT: Need to pip install vllm on a linux platform and run it there



from time import time
import pandas as pd
import numpy as np

from vllm import LLM, SamplingParams

from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

import re
import json
from json import JSONDecodeError

# path to model on the desktop
model_path = "/home/kuanpern/repos/Meta-Llama-3-8B-Instruct-hf-AWQ"


def create_template(article):
    '''Function creates a template to prompt the llm with, given an input of article content'''

    template = f'''
    <|begin_of_text|>
    <|start_header_id|>user<|end_header_id|>
    I have a news article i want to summarize. Help me summarize it in a paragraph, not more than 60 words.
    Your summary should be clear and concise, and must contain information on the purpose of the article, and the tone the article is written in.
    Check that your output satisfies the above.

    Example output (follow this template strictly):
    {{"summary": "Here is a summary of the news article"}}

    Your output should abide strictly by the format instructions given above. Do not output other text other than the JSON output with the summary in it. Do not give me code. Return only one output.
    Do not send the news article back to me. Do not send me any empty text. Do not send me multiple summaries. In the summary, if you need to use quotation marks, use only ''. Do not use "" as quotation marks.
    Here is the content of the news article: {article}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    '''

    return template


def parse_output(output):
    '''
    Input:
        - The input should be the llm output in text. We expect the text output to contain
        some form of JSON
    Output:
        - The output should be a 
    '''
    try:
        # first, try to parse the output directly
        output = json.loads(output)
        return output
    except JSONDecodeError:
        try:
            # if the output does not just contain purely the JSON string, look for the {} brackets to identify the JSON string in the output
            pattern = r'\{.*\}'
            new_output = re.search(pattern, output).group(0)
            new_output = json.loads(new_output)
            return new_output
        except:
            # if there are unescaped quotation marks (") in summary
            match = re.search(r'{"summary": "(.*)"}', output)
            if match:
                summary = match.group(1)
                escaped_summary = '"' + summary[1:-1].replace('"', '\\"') + '"'
                new_json_str = f'{{"summary": {escaped_summary}}}'
                return json.loads(new_json_str)
            else:
                print('JSON parsing error')


def write_to_json(generated_text, idx):
    '''
    Functionality: Writes to a JSON file, given a text in the form of a python dictionary, and the index of the file
    
    Inputs:
        - generated_text: text that is generated and already parsed by parse_output()
        - idx: index of the row in the dataframe read
    
    Output:
        - None
        - Writes to a .json file. The .json file contains the index inputted as well.
    '''
    with open(f'llama3_summarized_{idx}.json', 'w') as f:
        json.dump(generated_text, f)


def batch_generate(llm, idx, batch_size):
    '''
    Functionality:
        - Given some llm and some index, batch generate 100 responses
        - should be for text from idx to idx + batch_size

    Inputs:
        - llm: an LLM instance from the vllm package
        - idx: starting index of the dataframe
        - batch_size: size of batch we want to take for llm to process
    
    Outputs:
        - None
        - calls the write_to_json() function to write the outputs as json files
    '''
    prompts = [create_template(df['body'][i]) for i in range(idx, min(idx+batch_size, len(df)))] # usually contains batch_size number of prompts.
    # otherwise, if we exceed the index of df, we take len(df) instead as the endpoint

    # get llm to generate responses for the batch
    outputs = llm.generate(prompts, sampling_params)

    # parse the output and write to a new json file
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_text = parse_output(generated_text)

        # write generated text to a .json file
        write_to_json(generated_text, idx)
        idx += 1


if __name__ == '__main__':
    # read csv file
    df = pd.read_csv('compiled_data.csv')
    print('Dataframe has been loaded')

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.4, max_tokens=2000, stop_token_ids=['<|eot_id|>'])

    # Create an LLM.
    llm = LLM(model=model_path, gpu_memory_utilization=0.5)

    print("Starting text generation and file creation")
    start_time_regular = time()

    for i in range(0, len(df), 500):
        # using a batch size of 500
        batch_generate(llm, i, 500)

    duration_regular = time() - start_time_regular
    print(f'Generation took {duration_regular} seconds.')