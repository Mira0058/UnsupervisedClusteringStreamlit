import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st

def plot_elbow_method(scores_df):
    """
    Plot the Elbow Method graph
    
    Args:
        scores_df (pandas.DataFrame): DataFrame with cluster numbers and WCSS scores
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(scores_df['clusters'], scores_df['wcss'], marker='o')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS Score')
    ax.set_title('Elbow Method')
    ax.grid(True)
    return fig

def plot_silhouette_scores(scores_df):
    """
    Plot the Silhouette Scores graph
    
    Args:
        scores_df (pandas.DataFrame): DataFrame with cluster numbers and silhouette scores
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(scores_df['clusters'], scores_df['silhouette_score'], marker='o')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Scores')
    ax.grid(True)
    return fig

def plot_cluster_scatter(df, feature_x, feature_y, cluster_col='Cluster'):
    """
    Plot a scatter plot of clusters for two features
    
    Args:
        df (pandas.DataFrame): DataFrame with cluster assignments
        feature_x (str): Name of the feature for x-axis
        feature_y (str): Name of the feature for y-axis
        cluster_col (str): Name of the cluster column
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x=feature_x, 
        y=feature_y, 
        hue=cluster_col, 
        data=df, 
        palette='colorblind',
        ax=ax
    )
    ax.set_title(f'Clusters by {feature_x} and {feature_y}')
    return fig

def plot_pairplot(df, features, cluster_col='Cluster'):
    """
    Create a pairplot for multiple features
    
    Args:
        df (pandas.DataFrame): DataFrame with cluster assignments
        features (list): List of features to include in pairplot
        cluster_col (str): Name of the cluster column
    """
    plot_data = df[features + [cluster_col]].copy()
    fig = sns.pairplot(plot_data, hue=cluster_col, palette='colorblind')
    fig.fig.suptitle('Pairplot of Features by Cluster', y=1.02)
    return fig

def plot_cluster_profiles(df, features, cluster_col='Cluster'):
    """
    Plot cluster profiles (average feature values for each cluster)
    
    Args:
        df (pandas.DataFrame): DataFrame with cluster assignments
        features (list): List of features to include in profiles
        cluster_col (str): Name of the cluster column
    """
    # Calculate mean values of features for each cluster
    cluster_profiles = df.groupby(cluster_col)[features].mean()
    
    # Normalize the values for better visualization
    normalized_profiles = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    normalized_profiles.T.plot(kind='bar', ax=ax)
    ax.set_title('Cluster Profiles (Normalized Feature Values)')
    ax.set_xlabel('Features')
    ax.set_ylabel('Normalized Value')
    ax.legend(title='Cluster')
    plt.tight_layout()
    return fig

def plot_3d_clusters(df, x, y, z, cluster_col='Cluster'):
    """
    Create a 3D scatter plot of clusters
    
    Args:
        df (pandas.DataFrame): DataFrame with cluster assignments
        x (str): Feature for x-axis
        y (str): Feature for y-axis
        z (str): Feature for z-axis
        cluster_col (str): Name of the cluster column
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    clusters = df[cluster_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    
    for cluster, color in zip(clusters, colors):
        cluster_data = df[df[cluster_col] == cluster]
        ax.scatter(
            cluster_data[x],
            cluster_data[y],
            cluster_data[z],
            s=50,
            color=color,
            label=f'Cluster {cluster}'
        )
    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(f'3D Cluster Plot: {x} vs {y} vs {z}')
    ax.legend()
    
    return fig

def plot_cluster_distribution(df, cluster_col='Cluster'):
    """
    Plot distribution of samples across clusters
    
    Args:
        df (pandas.DataFrame): DataFrame with cluster assignments
        cluster_col (str): Name of the cluster column
    """
    cluster_counts = df[cluster_col].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Customer Distribution Across Clusters')
    
    # Add count labels above bars
    for i, count in enumerate(cluster_counts):
        ax.text(i, count + 1, str(count), ha='center')
    
    return fig