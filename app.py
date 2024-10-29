import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
players = pd.read_csv("Updated_Mall_Customers_with_Clusters.csv")

st.title("Customer Segmentation App")

# Input for new data point
age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Annual Income (k$)", min_value=0, max_value=150)
score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100)

# Fit KMeans model using existing clusters (if not already done)
if 'kmeans' not in st.session_state:
    features = players[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=5)  # Adjust based on your data
    kmeans.fit(features_scaled)
    st.session_state.kmeans = kmeans
    st.session_state.scaler = scaler

if st.button("Predict Cluster"):
    # Prepare new data point for prediction
    new_data_point = pd.DataFrame([[age, income, score]], columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"])
    
    # Normalize the new data point
    new_data_scaled = st.session_state.scaler.transform(new_data_point)
    
    # Predict the cluster
    closest_cluster = st.session_state.kmeans.predict(new_data_scaled)[0]
    
    # Display the predicted cluster
    st.write(f"Predicted Cluster: {closest_cluster}")
