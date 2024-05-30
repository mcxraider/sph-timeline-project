import json
import logging

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
    with open(output_path, 'w', encoding='utf-8') as fin:
        fin.write(json_data)
    logging.info(f"Enhanced timeline saved to '{output_path}'")
