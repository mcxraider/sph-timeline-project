import ast
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster


def scale_df_embeddings(df_train, df_test):
    print("Processing embedding data and scaling data...\n")
    #Deserializing the embeddings
    body_embeddings_train = np.array(df_train['embeddings'].apply(ast.literal_eval).tolist())
    title_embeddings_train = np.array(df_train['Title_embeddings'].apply(ast.literal_eval).tolist())
    tags_embeddings_train = np.array(df_train['tags_embeddings'].apply(ast.literal_eval).tolist())

    body_embeddings_test = np.array(df_test['embeddings'].apply(ast.literal_eval).tolist())
    title_embeddings_test = np.array(df_test['Title_embeddings'].apply(ast.literal_eval).tolist())
    tags_embeddings_test = np.array(df_test['tags_embeddings'].apply(ast.literal_eval).tolist())
    
    # Combine embeddings
    all_embeddings_train = np.concatenate((body_embeddings_train, title_embeddings_train, tags_embeddings_train), axis=1)
    all_embeddings_test = np.concatenate((body_embeddings_test, title_embeddings_test, tags_embeddings_test), axis=1)

    # Standardize embeddings
    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(all_embeddings_train)
    test_embeddings = scaler.transform(all_embeddings_test)
    return train_embeddings,  test_embeddings

def get_variance_performance(train_embeddings):
# Experiment for this variance range of 92% to 95%
    print("Finding best Model parameters...\n")
    variance_range = [0.92]
    #variance_range = list(np.arange(0.92, 0.95, 0.01))
    variance_dic = {}

    for variance in variance_range:
        pca = PCA(n_components=variance)
        train_pca_embeddings = pca.fit_transform(train_embeddings)
        
        # Range of max_d values to try, for this dataset we use 65
        max_d_values = np.arange(52, 58)
        
        # List to store silhouette scores
        silhouette_scores_train = []

        # Perform hierarchical clustering
        Z = linkage(train_pca_embeddings, method='ward')

        for max_d in max_d_values:
            clusters_train = fcluster(Z, max_d, criterion='distance')
            
            # Calculate silhouette score only if there are at least 2 unique clusters and fewer than the number of samples
            if 1 < len(set(clusters_train)) < len(train_pca_embeddings):
                score_train = silhouette_score(train_pca_embeddings, clusters_train)
            else:
                score_train = -1  # Assign a score of -1 if less than 2 unique clusters or too many clusters
            
            silhouette_scores_train.append(score_train)

        # Determine the best max_d
        best_max_d_train = max_d_values[np.argmax(silhouette_scores_train)]
        variance_dic[variance] = {
            'max_d_train': best_max_d_train,
            'best_train_silhouette': max(silhouette_scores_train)
        }
    return variance_dic

def get_best_variance(perf_results):
    highest_train_sil = 0
    best_variance_s = []
    for variance, scores in perf_results.items():
        if scores['best_train_silhouette'] > highest_train_sil:
            highest_train_sil = scores['best_train_silhouette']
            best_variance_s = [variance]  
        elif scores['best_train_silhouette'] == highest_train_sil:
            best_variance_s.append(variance)  
    
    final_best_max_d = perf_results[best_variance_s[0]]['max_d_train']
    print(f"Best variance for this dataset is {round(best_variance_s[0], 2)} and the best maximum distance is {final_best_max_d}\n")
    return round(best_variance_s[0], 2), final_best_max_d

def predict_cluster(test_embedding, train_embeddings, clusters):
        distances = np.linalg.norm(train_embeddings - test_embedding, axis=1)
        return clusters[np.argmin(distances)]

def get_cluster_labels(best_variance, best_max_d, train_embeddings, test_embeddings, df_train, df_test):
    # Perform PCA
    print(f"Training new Hierarchical Clustering model with best variance: {best_variance} and max_d: {best_max_d}\n")
    pca = PCA(n_components=best_variance)
    pca_train_embeddings = pca.fit_transform(train_embeddings)
    pca_test_embeddings = pca.transform(test_embeddings)

    Z = linkage(pca_train_embeddings, method='ward', metric='euclidean')
    clusters_train = fcluster(Z, best_max_d, criterion='distance')
    # Predict clusters for test data using the nearest cluster center

    test_clusters = [predict_cluster(te, pca_train_embeddings, clusters_train) for te in pca_test_embeddings]

    df_train['Cluster_labels'] = clusters_train
    df_test['Cluster_labels'] = test_clusters
    df_test.reset_index(drop=True, inplace=True)
    
    # Create a dictionary to store the results
    cluster_dict = {}

    # Populate the dictionary with cluster contents for each test point
    for i, (test_point, test_cluster) in enumerate(zip(df_test.itertuples(), test_clusters)):
        cluster_contents = []
        
        cluster_indices = np.where(clusters_train == test_cluster)[0]
        cluster_df = df_train.iloc[cluster_indices]
        
        cluster_dict = {
            "Test point": {'id': test_point.st_id,
                        "Title": test_point.Title, 
                        "Tags": test_point.tags},
            "Cluster": test_cluster,
            "Cluster contents": cluster_contents
        }
        
        for _, row in cluster_df.iterrows():
            cluster_contents.append({"id": row['st_id'], 
                                    "Title": row['Title'],
                                    "Tags": row['tags'], 
                                    })

    print(f"Cluster Label {test_cluster} is chosen\n")
    input_list = ""
    input_list += f"Test Artice Chosen: (Title: {cluster_dict['Test point']['Title']}\nTags: {cluster_dict['Test point']['Tags']}):\n"
    for _, row in cluster_df.iterrows():
        input_list += f"Article id: {row['st_id']}, Title: {row['Title']}, Tags: {row['tags']}]\n"
    return input_list, df_train, df_test

def generate_clusters(df_train, df_test):
    train_embeddings, test_embeddings = scale_df_embeddings(df_train, df_test)
    variance_perf = get_variance_performance(train_embeddings)
    best_variance, best_max_d = get_best_variance(variance_perf)
    relevant_articles, df_train, df_test = get_cluster_labels(best_variance, best_max_d, train_embeddings, test_embeddings, df_train, df_test)
    return relevant_articles, df_train, df_test
