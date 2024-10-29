# app.py
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# Load dataset
players = pd.read_csv("Customers_Spending_with_Clusters.csv")

st.title("Customer Spending App")

# Input for new data point
age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Annual Income (k$)", min_value=0, max_value=150)
score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100)

if st.button("Predict Cluster"):
    new_data_point = pd.DataFrame([[age, income, score]], columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"])
    # Perform normalization and find cluster, then display result
    st.write(f"Predicted Cluster: {closest_cluster}")
