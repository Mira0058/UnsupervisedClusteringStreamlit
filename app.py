import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Import custom modules
from src.data_processing import load_data, preprocess_data, find_optimal_clusters, create_clusters
from src.visualization import (plot_elbow_method, plot_silhouette_scores, 
                              plot_cluster_scatter, plot_pairplot, 
                              plot_cluster_profiles, plot_3d_clusters,
                              plot_cluster_distribution)
from src.model_training import train_model, load_model, predict_clusters
from src.utils import get_feature_description, describe_clusters, create_sample_data, ensure_dir

# Set page configuration
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# Ensure directories exist
ensure_dir("data")
ensure_dir("models")

# Title and description
st.title("🛍️ Mall Customer Segmentation")
st.markdown("""
This application performs customer segmentation using K-means clustering algorithm.
The sample dataset is used to identify distinct customer groups.
""")

# Sidebar
st.sidebar.header("Options")

# Create and use sample data
df = create_sample_data()

# Save sample data to disk
df.to_csv(os.path.join("data", "mall_customers.csv"), index=False)
st.sidebar.info("Using sample data.")

# Main content
# Preprocess data
df = preprocess_data(df)

# Display tabs
tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Feature Analysis", "Clustering", "Results"])

with tab1:
    st.header("Data Overview")
    
    # Display dataset info
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Display dataset statistics
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())
    
    # Display column information
    st.subheader("Column Information")
    for col in df.columns:
        st.write(f"**{col}**: {get_feature_description(col)}")

with tab2:
    st.header("Feature Analysis")
    
    # Select features for analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    # Pairplot of features
    st.subheader("Pairplot of Features")
    fig = sns.pairplot(df[numeric_cols])
    st.pyplot(fig)
    
    # Distribution of features
    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select feature to visualize:", numeric_cols)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[feature_to_plot], kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature_to_plot}")
    st.pyplot(fig)

with tab3:
    st.header("Clustering Analysis")
    
    # Select features for clustering
    st.subheader("Feature Selection")
    available_features = [col for col in numeric_cols if col != 'CustomerID']
    default_features = [feat for feat in ['Age', 'Annual_Income', 'Spending_Score'] if feat in available_features]
    
    selected_features = st.multiselect(
        "Select features for clustering:",
        available_features,
        default=default_features
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features for clustering.")
    else:
        # Find optimal number of clusters
        st.subheader("Finding Optimal Number of Clusters")
        
        max_clusters = st.slider("Maximum number of clusters to consider:", 2, 12, 8)
        
        with st.spinner("Finding optimal number of clusters..."):
            optimal_clusters, scores_df = find_optimal_clusters(df, selected_features, max_clusters)
            
            st.write(f"**Suggested optimal number of clusters:** {optimal_clusters}")
            
            # Plot elbow method
            st.subheader("Elbow Method")
            fig_elbow = plot_elbow_method(scores_df)
            st.pyplot(fig_elbow)
            
            # Plot silhouette scores
            st.subheader("Silhouette Scores")
            fig_silhouette = plot_silhouette_scores(scores_df)
            st.pyplot(fig_silhouette)
        
        # Select number of clusters
        n_clusters = st.number_input(
            "Select number of clusters:",
            min_value=2,
            max_value=12,
            value=optimal_clusters
        )
        
        # Train model and create clusters
        if st.button("Create Clusters"):
            with st.spinner("Creating clusters..."):
                # Train model
                model_path = os.path.join("models", "kmeans_model.pkl")
                kmeans_model = train_model(df, selected_features, n_clusters, model_path)
                
                # Create clusters
                clustered_df, cluster_centers = create_clusters(df, selected_features, n_clusters)
                
                # Save results to session state for tab4
                st.session_state.clustered_df = clustered_df
                st.session_state.cluster_centers = cluster_centers
                st.session_state.selected_features = selected_features
                
                st.success("Clusters created successfully! Go to the Results tab to view them.")

with tab4:
    st.header("Clustering Results")
    
    if 'clustered_df' not in st.session_state:
        st.info("Please create clusters in the Clustering tab first.")
    else:
        clustered_df = st.session_state.clustered_df
        cluster_centers = st.session_state.cluster_centers
        selected_features = st.session_state.selected_features
        
        # Display cluster centers
        st.subheader("Cluster Centers")
        st.dataframe(cluster_centers)
        
        # Display cluster distribution
        st.subheader("Cluster Distribution")
        fig_dist = plot_cluster_distribution(clustered_df)
        st.pyplot(fig_dist)
        
        # Display cluster profiles
        st.subheader("Cluster Profiles")
        fig_profiles = plot_cluster_profiles(clustered_df, selected_features)
        st.pyplot(fig_profiles)
        
        # Cluster descriptions
        st.subheader("Cluster Descriptions")
        descriptions = describe_clusters(clustered_df, selected_features)
        
        for cluster, desc in descriptions.items():
            with st.expander(f"Cluster {cluster}"):
                for point in desc:
                    st.write(f"- {point}")
        
        # Scatter plot visualization
        st.subheader("Cluster Visualization")
        
        if len(selected_features) == 2:
            # 2D visualization
            feature_x, feature_y = selected_features
            fig_scatter = plot_cluster_scatter(clustered_df, feature_x, feature_y)
            st.pyplot(fig_scatter)
        
        elif len(selected_features) >= 3:
            # 3D visualization
            st.write("3D visualization of the first three selected features:")
            feature_x, feature_y, feature_z = selected_features[:3]
            fig_3d = plot_3d_clusters(clustered_df, feature_x, feature_y, feature_z)
            st.pyplot(fig_3d)
            
            # 2D visualization options
            st.write("2D visualization of selected features:")
            col1, col2 = st.columns(2)
            with col1:
                feature_x = st.selectbox("X-axis feature:", selected_features)
            with col2:
                feature_y = st.selectbox("Y-axis feature:", [f for f in selected_features if f != feature_x])
            
            fig_scatter = plot_cluster_scatter(clustered_df, feature_x, feature_y)
            st.pyplot(fig_scatter)
        
        # Download clustered data
        st.subheader("Download Results")
        csv = clustered_df.to_csv(index=False)
        st.download_button(
            label="Download clustered data as CSV",
            data=csv,
            file_name="clustered_customers.csv",
            mime="text/csv"
        )

# Footer
st.markdown("""
---
### Mall Customer Segmentation Project
This application allows mall owners and marketers to segment customers based on their behavior and demographics.
""")