import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Cluster Descriptions
CLUSTER_DESCRIPTIONS = {
    -1: "🏦 Noise Points: Customers with unique or outlier characteristics",
    0: "🏦 Conservative Spenders (High Income): Stable customers with high income but moderate spending",
    1: "⚠️ Risk Customers: Potential high-risk or unpredictable spending pattern",
    2: "💎 Premium Customers: High-value customers with significant spending capacity",
    3: "⚖️ Balanced Group: Well-rounded customers with balanced financial behaviors"
}

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
    
    # Sidebar option to view full dataset
    if st.sidebar.button('📋 Full Dataset'):
        st.subheader('Full Customer Dataset')
        st.dataframe(data)
    
    # Sidebar option to view cluster descriptions
    if st.sidebar.button('📊 Cluster Descriptions'):
        st.subheader('Cluster Descriptions')
        for cluster, description in CLUSTER_DESCRIPTIONS.items():
            st.write(f"**Cluster {cluster}**: {description}")
    
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
        
        # Show cluster with description and emoji
        cluster_description = CLUSTER_DESCRIPTIONS.get(predicted_cluster, "Unknown Cluster")
        st.write(f'Predicted Cluster: {predicted_cluster} - {cluster_description}')
        
        # Additional context about the cluster
        if predicted_cluster != -1:
            cluster_data = clustered_data[clustered_data['Cluster'] == predicted_cluster]
            st.write('Cluster Characteristics:')
            st.write(cluster_data[['Age', 'Income (INR)', 'Spending  (1-100)', 'Gender']].describe())
        else:
            st.write('This data point is considered an outlier (Noise) by the DBSCAN algorithm.')

if __name__ == '__main__':
    main()
