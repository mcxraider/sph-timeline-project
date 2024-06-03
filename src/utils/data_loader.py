import json
import logging
import pandas as pd
from typing import Optional


def load_single_json(file_path: str) -> Optional[dict]:
    """
    Load JSON data from a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
            print("Files Loaded")
        logging.info(f"JSON file '{file_path}' loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error(f"File '{file_path}' not found. Please check the file path.")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file '{file_path}'. Please check the file content.")
        return None

def combine_4_json(files):
    combined_data = []
    for file in files:
        with open(file, 'r') as f:
            # Load data from the file and append it to the combined list
            data = json.load(f)
            combined_data.extend(data)
    print("Input files combined\n")
    return combined_data

def read_load_json_to_df(json_data):
    for item in json_data:
        #Convert the embeddings to json string as CSVs dont accept list as a data type
        item['tags_embeddings'] = json.dumps(item['tags_embeddings'])
        item['Title_embeddings'] = json.dumps(item['Title_embeddings'])
    df = pd.DataFrame(json_data)
    print("Input data converted and read in\n")
    return df

def load_df(files):
    db = combine_4_json(files)
    df = read_load_json_to_df(db)
    #Drop nan rows 
    final_df = df.drop(df[df.isnull().any(axis=1)].index)
    return final_df


def split_batches(timeline, max_batch_size=30):
    n = len(timeline)
    if n <= max_batch_size:
        return [timeline]
    
    num_batches = n // max_batch_size
    remainder = n % max_batch_size
    
    if remainder > 0 and remainder < max_batch_size // 2:
        num_batches -= 1
        remainder += max_batch_size

    batches = []
    start = 0
    for i in range(num_batches):
        end = start + max_batch_size
        batches.append(timeline[start:end])
        start = end
    
    if remainder > 0:
        batches.append(timeline[start:start + remainder])
    
    return batches