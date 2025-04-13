import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data(file_path):
    """
    Load the mall customer dataset
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocess the data for clustering
    
    Args:
        df (pandas.DataFrame): Raw dataframe
        
    Returns:
        pandas.DataFrame: Preprocessed dataframe
    """
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    
    # Ensure column names match our expected format
    expected_columns = ['CustomerID', 'Gender', 'Age', 'Annual_Income', 'Spending_Score']
    
    # Rename columns if needed
    column_mapping = {}
    if 'Annual Income (k$)' in df.columns:
        column_mapping['Annual Income (k$)'] = 'Annual_Income'
    if 'Spending Score (1-100)' in df.columns:
        column_mapping['Spending Score (1-100)'] = 'Spending_Score'
        
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    return df

def find_optimal_clusters(df, features, max_clusters=8):
    """
    Find the optimal number of clusters using silhouette score
    
    Args:
        df (pandas.DataFrame): Preprocessed dataframe
        features (list): List of feature columns to use for clustering
        max_clusters (int): Maximum number of clusters to try
        
    Returns:
        tuple: (optimal clusters, dataframe with scores)
    """
    k_range = range(2, max_clusters + 1)
    K = []
    silhouette_scores = []
    wcss_scores = []
    
    for k in k_range:
        # Create and fit KMeans model
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(df[features])
        
        # Get cluster labels
        labels = kmeans.labels_
        
        # Calculate silhouette score
        sil_score = silhouette_score(df[features], labels)
        silhouette_scores.append(sil_score)
        
        # Calculate WCSS (Within-Cluster Sum of Square)
        wcss = kmeans.inertia_
        wcss_scores.append(wcss)
        
        K.append(k)
    
    # Create dataframe with scores
    scores_df = pd.DataFrame({
        'clusters': K,
        'silhouette_score': silhouette_scores,
        'wcss': wcss_scores
    })
    
    # Find optimal number of clusters (highest silhouette score)
    optimal_clusters = K[silhouette_scores.index(max(silhouette_scores))]
    
    return optimal_clusters, scores_df

def create_clusters(df, features, n_clusters):
    """
    Create clusters using KMeans
    
    Args:
        df (pandas.DataFrame): Preprocessed dataframe
        features (list): List of feature columns to use for clustering
        n_clusters (int): Number of clusters
        
    Returns:
        pandas.DataFrame: Dataframe with cluster assignments
    """
    # Create a copy of the dataframe
    clustered_df = df.copy()
    
    # Create and fit KMeans model
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(clustered_df[features])
    
    # Add cluster labels to dataframe
    clustered_df['Cluster'] = kmeans.labels_
    
    # Get cluster centers
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=features
    )
    
    return clustered_df, cluster_centers