import logging
import pandas as pd
import sys
from utils.data_loader import *
from utils.timeline_enhancer import *
from utils.json_utils import *
from utils.heir_clustering import *



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    #Could i possibly do this in another python file?
    files = ['../data_upload/final_db1.json', '../data_upload/final_db2.json', '../data_upload/final_db3.json', '../data_upload/final_db4.json']
    db = combine_4_json(files)
    df = read_load_json_to_df(db)
    #Drop nan rows 
    df = df.drop(df[df.isnull().any(axis=1)].index)
    
    
    X_train_scaled, X_test_scaled = split_scale_embeddings(df)
    variance_perf = get_variance_perf(X_train_scaled, X_test_scaled)
    best_variance, best_max_d = get_best_variance(variance_perf)
    
    df_train, df_test = split_scale_df(df)
    train_embeddings, test_embeddings = scale_df_embeddings(df_train, df_test)
    train_clusters, test_clusters = get_cluster_labels(best_variance, best_max_d, train_embeddings,  test_embeddings)
    similar_articles_dict = get_similar_articles(train_clusters, test_clusters, df_train, df_test)
    
    
    
    
    
    
    
    
    single_timeline = '../data_upload/single_timeline_trial.json'
    single_timeline_articles = '../data_upload/df_retrieve.json'
    output_path = '../data_upload/enhanced_timeline_trial.json'

    timeline_data = load_single_json(single_timeline)
    if timeline_data is None:
        logging.warning("No timeline data to merge.")
        sys.exit()

    article_data = load_single_json(single_timeline_articles)
    if article_data is None:
        logging.warning("No data loaded for retrieval, check file")
        sys.exit()

    retrieval = pd.DataFrame(article_data)

    sorted_timeline = first_timeline_merge(timeline_data)
    print("First timeline enhancement done")
    final_timeline = second_timeline_enhancement(sorted_timeline, retrieval)
    print("Second timeline enhancement done")
    save_enhanced_timeline(final_timeline, output_path)

if __name__ == "__main__":
    main()
    