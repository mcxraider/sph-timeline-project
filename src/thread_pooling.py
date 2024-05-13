import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import trange


def load_and_merge_csv(file_pattern, num_files):
    file_names = [file_pattern.format(i) for i in range(1, num_files + 1)]
    dataframes = [pd.read_csv(filename) for filename in file_names]
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df

def fetch_tags(article_pair):
    article_text, article_id = article_pair
    time.sleep(1)  
    return article_id, ["tag1", "tag2", "tag3"]

def process_articles(df, max_workers=10):
    results = {}
    batch_size = 100
    cooldown_period = 10  

    articles = df['combined'].tolist()
    ids = df['id'].tolist()
    article_id_pairs = list(zip(articles, ids))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(article_id_pairs), batch_size):
            current_batch = article_id_pairs[i:i+batch_size]
            print(f"Starting batch processing for articles {i+1} to {min(i+batch_size, len(article_id_pairs))}")
            futures = {executor.submit(fetch_tags, pair): pair for pair in current_batch}

            processed_count = i
            for future in as_completed(futures):
                article_id, tags = future.result()
                results[article_id] = tags
                processed_count += 1
                print(f"Processed article {processed_count} in Batch {(i//100)+1} ")

            if processed_count >= len(article_id_pairs):
                return results

            print(f"All tasks in batch {i//batch_size + 1} completed, cooling down for {cooldown_period} seconds...")
            time.sleep(cooldown_period)
    return results

df = load_and_merge_csv('../data_upload/cluster_labels{}.csv', 4)
df = df.loc[range(200)]
tags = process_articles(df)
