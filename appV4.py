import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def load_data():
    """Load the customer dataset"""
    return pd.read_csv('Customers Dataset DBSCAN With Cluster.csv')

def prepare_clustering_model(data):
    """
    Prepare the clustering model using selected features
    
    Returns:
    - Scaled features
    - Scaler object
    - DBSCAN model
    """
    # Select features for clustering
    features = ['CustomerID', 'Gender', 'Age', 'Income (INR)', 'Spending  (1-100)']
    
    # Preprocess the data
    # Convert Gender to numeric
    gender_map = {'Male': 0, 'Female': 1}
    data['Gender_Numeric'] = data['Gender'].map(gender_map)
    
    # Select features for clustering (including numeric gender)
    X = data[['Age', 'Income (INR)', 'Spending  (1-100)', 'Gender_Numeric']]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    data['Cluster'] = dbscan.fit_predict(X_scaled)
    
    return X_scaled, scaler, dbscan, data

def predict_cluster(input_data, scaler, dbscan_model):
    """
    Predict the cluster for input data
    
    Parameters:
    - input_data: Input features
    - scaler: Fitted StandardScaler
    - dbscan_model: Trained DBSCAN model
    
    Returns:
    - Predicted cluster
    """
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Predict cluster
    predicted_cluster = dbscan_model.fit_predict(input_scaled)
    
    return predicted_cluster[0]

def main():
    st.title('Customer Cluster Predictor')
    
    # Load and prepare data
    data = load_data()
    _, scaler, dbscan_model, clustered_data = prepare_clustering_model(data)
    
    # Sidebar navigation
    menu = st.sidebar.selectbox('Menu', 
        ['Cluster Prediction', 'View Dataset', 'Cluster Summary'])
    
    if menu == 'Cluster Prediction':
        # Input form
        st.header('Enter Customer Details')
        
        # Input fields
        customer_id = st.number_input('Customer ID', min_value=1, value=1)
        
        # Gender selection
        gender = st.selectbox('Gender', ['Male', 'Female'])
        
        # Numeric inputs
        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        income = st.number_input('Income (INR)', min_value=1000, max_value=1000000, value=15000)
        spending = st.number_input('Spending (1-100)', min_value=1, max_value=100, value=50)
        
        # Predict button
        if st.button('Predict Cluster'):
            # Prepare input data
            gender_numeric = 0 if gender == 'Male' else 1
            input_data = np.array([[age, income, spending, gender_numeric]])
            
            # Predict cluster
            predicted_cluster = predict_cluster(input_data, scaler, dbscan_model)
            
            # Display results
            st.subheader('Clustering Result')
            st.write(f'Predicted Cluster: {predicted_cluster}')
            
            # Additional context about the cluster
            if predicted_cluster != -1:
                cluster_data = clustered_data[clustered_data['Cluster'] == predicted_cluster]
                st.write('Cluster Characteristics:')
                st.write(cluster_data[['Age', 'Income (INR)', 'Spending  (1-100)', 'Gender']].describe())
            else:
                st.write('This data point is considered an outlier (Noise) by the DBSCAN algorithm.')
    
    elif menu == 'View Dataset':
        st.header('Customer Dataset')
        
        # Filtering options
        st.sidebar.header('Dataset Filters')
        show_columns = st.sidebar.multiselect(
            'Select Columns to Display', 
            data.columns.tolist(), 
            default=['CustomerID', 'Name', 'Gender', 'Age', 'Income (INR)', 'Cluster']
        )
        
        # Filter by cluster
        unique_clusters = sorted(data['Cluster'].unique())
        selected_clusters = st.sidebar.multiselect(
            'Filter by Cluster', 
            unique_clusters, 
            default=unique_clusters
        )
        
        # Apply filters
        filtered_data = data[
            (data['Cluster'].isin(selected_clusters)) &
            (show_columns)
        ]
        
        # Display filtered dataset
        st.dataframe(filtered_data[show_columns])
        
        # Basic dataset stats
        st.subheader('Dataset Statistics')
        st.write(filtered_data.describe())
    
    elif menu == 'Cluster Summary':
        st.header('Cluster Summary')
        
        # Cluster distribution
        cluster_distribution = clustered_data['Cluster'].value_counts()
        st.subheader('Cluster Distribution')
        st.write(cluster_distribution)
        
        # Detailed cluster characteristics
        st.subheader('Cluster Characteristics')
        cluster_summary = clustered_data.groupby('Cluster')[
            ['Age', 'Income (INR)', 'Spending  (1-100)']
        ].agg(['mean', 'min', 'max'])
        st.write(cluster_summary)
        
        # Gender distribution per cluster
        st.subheader('Gender Distribution per Cluster')
        gender_distribution = clustered_data.groupby(['Cluster', 'Gender']).size().unstack(fill_value=0)
        st.write(gender_distribution)

if __name__ == '__main__':
    main()
