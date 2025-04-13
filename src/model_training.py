from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import pickle
import os

def train_model(df, features, n_clusters, model_path=None):
    """
    Train a KMeans model and save it
    
    Args:
        df (pandas.DataFrame): Preprocessed dataframe
        features (list): List of feature columns to use for clustering
        n_clusters (int): Number of clusters
        model_path (str): Path to save the model
        
    Returns:
        KMeans: Trained KMeans model
    """
    # Create and fit KMeans model
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(df[features])
    
    # Save the model if path is provided
    if model_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(kmeans, f)
    
    return kmeans

def load_model(model_path):
    """
    Load a trained KMeans model
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        KMeans: Loaded KMeans model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def predict_clusters(model, data):
    """
    Predict clusters for new data
    
    Args:
        model (KMeans): Trained KMeans model
        data (pandas.DataFrame): New data to predict clusters for
        
    Returns:
        numpy.ndarray: Predicted cluster labels
    """
    return model.predict(data)