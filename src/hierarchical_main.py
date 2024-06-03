import logging
import yaml
from utils.data_loader import load_df
from utils.hierarchical_clustering import *
from utils.timeline_generator import *


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(config_path='../config.yaml'):
    config = load_config(config_path)
    files = config['data_input_files']
    output_path = config['output_path']
    
    df = load_df(files)
    df_train, df_test = split_df(df)

    #check if the test point is worth generating a timeline. 
    if not to_generate_timeline(df_test):
        return None, None
    
    relevant_articles, df_train, df_test = generate_clusters(df_train, df_test)
    final_timeline = generate_save_timeline(relevant_articles, df_train, df_test, output_path)
    return final_timeline
    
if __name__ == "__main__":
    final_timeline = main()
