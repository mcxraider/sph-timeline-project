import logging
import yaml
from utils.data_loader import load_df
from utils.json_utils import *
from utils.heir_clustering import *
from utils.timeline_generator import *


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    files = ['../data_upload/final_db1.json', '../data_upload/final_db2.json', '../data_upload/final_db3.json', '../data_upload/final_db4.json']
    df = load_df(files)
    df_train, df_test = split_df(df)

    #check if the test point is worth generating a timeline. 
    if not to_generate_timeline(df_test):
        return None, None
    
    relevant_articles, df_train, df_test = generate_cluster(df_train, df_test)
    final_timeline = generate_save_timeline(relevant_articles, df_train, df_test)
    return final_timeline, retrieval
    
if __name__ == "__main__":
    final_timeline, retrieval = main()
