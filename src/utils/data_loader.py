import json
import logging
import ast
from typing import Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_single_json(file_path: str) -> Optional[dict]:
    """
    Load JSON data from a file.
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


def combine_4_json(files):
    combined_data = []
    for file in files:
        with open(file, 'r') as f:
            # Load data from the file and append it to the combined list
            data = json.load(f)
            combined_data.extend(data)
    return combined_data



def read_load_json_to_df(json_data):
    for item in json_data:
        #Convert the embeddings to json string as CSVs dont accept list as a data type
        item['tags_embeddings'] = json.dumps(item['tags_embeddings'])
        item['Title_embeddings'] = json.dumps(item['Title_embeddings'])
    df = pd.DataFrame(json_data)
    return df




    