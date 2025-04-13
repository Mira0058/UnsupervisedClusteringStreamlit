import pandas as pd
import streamlit as st
import os

def get_feature_description(feature):
    """
    Get description for features
    
    Args:
        feature (str): Name of the feature
        
    Returns:
        str: Description of the feature
    """
    descriptions = {
        'CustomerID': 'Unique ID assigned to the customer',
        'Gender': 'Gender of the customer',
        'Age': 'Age of the customer',
        'Annual_Income': 'Annual Income of the customers in 1000 dollars',
        'Spending_Score': 'Score assigned between 1-100 by the mall based on customer spending behavior'
    }
    
    return descriptions.get(feature, 'No description available')

def describe_clusters(df, features, cluster_col='Cluster'):
    """
    Generate descriptions for each cluster based on their average feature values
    
    Args:
        df (pandas.DataFrame): DataFrame with cluster assignments
        features (list): List of feature columns used for clustering
        cluster_col (str): Name of the cluster column
        
    Returns:
        dict: Dictionary with cluster descriptions
    """
    # Calculate mean values for each feature in each cluster
    cluster_means = df.groupby(cluster_col)[features].mean()
    
    # Overall means for comparison
    overall_means = df[features].mean()
    
    # Create descriptions
    descriptions = {}
    
    for cluster in sorted(df[cluster_col].unique()):
        description = []
        
        for feature in features:
            cluster_value = cluster_means.loc[cluster, feature]
            overall_value = overall_means[feature]
            
            # Determine if value is high, medium, or low compared to overall
            if cluster_value > overall_value * 1.1:
                level = "high"
            elif cluster_value < overall_value * 0.9:
                level = "low"
            else:
                level = "average"
            
            description.append(f"{feature.replace('_', ' ')}: {level} ({cluster_value:.1f})")
        
        descriptions[cluster] = description
    
    return descriptions

def create_sample_data():
    """
    Create a small sample dataset similar to mall customers data
    
    Returns:
        pandas.DataFrame: Sample dataset
    """
    data = {
        'CustomerID': list(range(1, 11)),
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Age': [23, 45, 35, 28, 55, 19, 42, 31, 38, 51],
        'Annual_Income': [25, 45, 35, 80, 60, 30, 70, 50, 40, 90],
        'Spending_Score': [30, 75, 50, 90, 20, 85, 40, 60, 45, 10]
    }
    
    return pd.DataFrame(data)

def ensure_dir(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)