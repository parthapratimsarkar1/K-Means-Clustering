import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time

# Load dataset
players = pd.read_csv("Updated_Mall_Customers_with_Clusters.csv")

# Assuming the dataset has 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)' columns
X = players[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Fit the KMeans model
kmeans = KMeans(n_clusters=3)  # Change the number of clusters as needed
kmeans.fit(X_normalized)

# Define cluster specifications (example data, modify as needed)
cluster_specs = {
    0: "Cluster 0: Young customers with low spending scores.",
    1: "Cluster 1: Middle-aged customers with high income.",
    2: "Cluster 2: Older customers with average spending scores.",
    3: "Cluster 3: Young customers with high spending scores.",
    4: "Cluster 4: All-age customers with varying incomes."
}

st.title("Customer Segmentation App")

# Input for new data point
age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Annual Income (k$)", min_value=0, max_value=150)
score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100)

if st.button("Predict Cluster"):
    with st.spinner("Predicting..."):
        # Simulate a delay for the animation effect
        time.sleep(2)
        
        new_data_point = pd.DataFrame([[age, income, score]], columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"])
        
        # Normalize the new data point
        new_data_normalized = scaler.transform(new_data_point)
        
        # Perform clustering
        closest_cluster = kmeans.predict(new_data_normalized)[0]
        
        # Display the result
        st.success(f"Predicted Cluster: {closest_cluster}")
        st.write(cluster_specs[closest_cluster])

