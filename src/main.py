import logging
import pandas as pd
import sys
from utils.data_loader import load_single_json
from utils.timeline_enhancer import first_timeline_merge, second_timeline_enhancement, clean_output
from utils.json_utils import save_enhanced_timeline


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
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
    
