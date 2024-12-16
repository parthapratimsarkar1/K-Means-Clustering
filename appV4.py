import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import seaborn as sns

def load_data():
    """Load the customer dataset"""
    return pd.read_csv('Customers Dataset DBSCAN With Cluster.csv')

def perform_clustering(data, features, eps, min_samples):
    """
    Perform DBSCAN clustering on selected features
    
    Parameters:
    - data: Input DataFrame
    - features: List of features to use for clustering
    - eps: Epsilon (neighborhood distance) parameter
    - min_samples: Minimum number of samples in a neighborhood
    
    Returns:
    - Clustered DataFrame
    - DBSCAN model
    """
    # Select features for clustering
    X = data[features]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    data['Cluster'] = dbscan.fit_predict(X_scaled)
    
    return data, dbscan

def plot_clusters(data, x_feature, y_feature):
    """
    Create a scatter plot of clusters
    
    Parameters:
    - data: Clustered DataFrame
    - x_feature: Feature for x-axis
    - y_feature: Feature for y-axis
    
    Returns:
    - Matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(data=data, x=x_feature, y=y_feature, hue='Cluster', palette='viridis')
    plt.title(f'Clustering Results: {x_feature} vs {y_feature}')
    return scatter.get_figure()

def main():
    st.title('Customer Clustering Analysis with DBSCAN')
    
    # Load data
    data = load_data()
    
    # Sidebar for input parameters
    st.sidebar.header('Clustering Parameters')
    
    # Feature selection
    available_features = ['Age', 'Income (INR)', 'Spending  (1-100)', 'CIBIL Score']
    selected_features = st.sidebar.multiselect(
        'Select Features for Clustering', 
        available_features, 
        default=['Age', 'Income (INR)']
    )
    
    # DBSCAN parameters
    eps = st.sidebar.slider('Epsilon (neighborhood distance)', 0.1, 2.0, 0.5, 0.1)
    min_samples = st.sidebar.slider('Minimum Samples', 1, 10, 3, 1)
    
    # Visualization features
    x_feature = st.sidebar.selectbox('X-axis Feature', selected_features, index=0)
    y_feature = st.sidebar.selectbox('Y-axis Feature', selected_features, index=1)
    
    # Perform clustering
    if len(selected_features) >= 2:
        clustered_data, dbscan_model = perform_clustering(data, selected_features, eps, min_samples)
        
        # Display clustering results
        st.subheader('Clustering Results')
        
        # Cluster distribution
        cluster_counts = clustered_data['Cluster'].value_counts()
        st.write('Cluster Distribution:')
        st.write(cluster_counts)
        
        # Visualization
        st.subheader('Cluster Visualization')
        fig = plot_clusters(clustered_data, x_feature, y_feature)
        st.pyplot(fig)
        
        # Detailed cluster information
        st.subheader('Cluster Details')
        cluster_details = clustered_data.groupby('Cluster')[selected_features].mean()
        st.write(cluster_details)
    else:
        st.warning('Please select at least 2 features for clustering')

if __name__ == '__main__':
    main()
