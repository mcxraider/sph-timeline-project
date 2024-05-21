import json
import logging
from typing import Optional

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
