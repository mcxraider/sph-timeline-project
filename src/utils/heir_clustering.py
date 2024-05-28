import os
import ast
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tqdm import trange
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster




def split_scale_embeddings(df):
    # Deserialising of embeddings
    body_embeddings= np.array(df['embeddings'].apply(ast.literal_eval).tolist())
    title_embeddings= np.array(df['Title_embeddings'].apply(ast.literal_eval).tolist())
    tags_embeddings= np.array(df['tags_embeddings'].apply(ast.literal_eval).tolist())
    all_embeddings = np.concatenate((body_embeddings, title_embeddings, tags_embeddings), axis=1)

    train_embeddings, test_embeddings = train_test_split(all_embeddings, test_size=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_embeddings)
    X_test_scaled = scaler.transform(test_embeddings)
        
    return X_train_scaled, X_test_scaled

    

def get_variance_perf(X_train_scaled, X_test_scaled):
# Experiment for this variance range of 94% to 97%
    variance_range = list(np.arange(0.92, 0.95, 0.01))
    variance_dic = {}

    for variance in variance_range:
        pca = PCA(n_components=variance)
        train_pca_embeddings = pca.fit_transform(X_train_scaled)
        
        # Range of max_d values to try
        max_d_values = np.arange(45, 70)
        
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
    print(f"Best variance for this clustering is {round(best_variance_s[0], 2)} and the best maximum distance is {final_best_max_d}")
    return round(best_variance_s[0], 2), final_best_max_d


def split_scale_df(df):
    df_test = df.sample(1)
    df_train = df.drop(df_test.index)
    return df_train, df_test

def scale_df_embeddings(df_train, df_test):
    # Deserializing the embeddings
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



def get_cluster_labels(best_variance, best_max_d, train_embeddings, test_embeddings):
    # Perform PCA
    pca = PCA(n_components=best_variance)
    pca_train_embeddings = pca.fit_transform(train_embeddings)
    pca_test_embeddings = pca.transform(test_embeddings)


    Z = linkage(pca_train_embeddings, method='ward', metric='euclidean')
    clusters_train = fcluster(Z, best_max_d, criterion='distance')
    def predict_cluster(test_embedding, train_embeddings, clusters):
        distances = np.linalg.norm(train_embeddings - test_embedding, axis=1)
        return clusters[np.argmin(distances)]

    # Assign clusters to test points
    test_clusters = [predict_cluster(te, pca_train_embeddings, clusters_train) for te in pca_test_embeddings]
    return clusters_train, test_clusters


def get_common_tags(tags1, tags2):
        return set(tags1).intersection(set(tags2))


def get_similar_articles(clusters_train, test_clusters, df_train, df_test):
    #Update train and test dataset
    df_train['Cluster_labels'] = clusters_train
    df_test['Cluster_labels'] = test_clusters
    df_test.reset_index(drop=True, inplace=True)

    # Initialize the dictionary to store similar articles
    similar_articles_dict = {}

    # Iterate over each test article in the filtered df_test
    for index, test_row in df_test.iterrows():
        test_tags = test_row['tags']
        test_cluster_label = test_row['Cluster_labels']
        
        # Filter df_train for the same cluster label
        df_train_cluster = df_train[df_train['Cluster_labels'] == test_cluster_label]
        
        # Find similar articles in df_train
        similar_indexes = []
        for train_index, train_row in df_train_cluster.iterrows():
            train_tags = train_row['tags']
            if len(get_common_tags(test_tags, train_tags)) >= 2:
                similar_indexes.append(train_index)
        
        # Store the result in the dictionary if there are at least 2 supporting articles
        if len(similar_indexes) >= 2:
            similar_articles_dict[index] = {
                'Title': test_row['Title'],
                'indexes': similar_indexes,
                'Text': test_row['Text']
            }
    return similar_articles_dict