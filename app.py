import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time

# Set up the Streamlit app title
st.title("Customer Segmentation App")

# File uploader to upload the CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset from the uploaded file
    players = pd.read_csv(uploaded_file)

    # Display the dataset
    st.subheader("Dataset Preview:")
    st.write(players.head())

    # Check if the required columns exist
    required_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    if all(col in players.columns for col in required_columns):
        # Prepare the data for clustering
        X = players[required_columns]

        # Normalize the data
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        # Fit the KMeans model
        n_clusters = st.number_input("Number of Clusters", min_value=1, max_value=10, value=5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_normalized)

        # Create cluster specifications
        cluster_specs = {i: f"Cluster {i}: Description based on analysis." for i in range(n_clusters)}

        # Input for new data point
        age = st.number_input("Age", min_value=18, max_value=100)
        income = st.number_input("Annual Income (k$)", min_value=0, max_value=150)
        score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100)

        if st.button("Predict Cluster"):
            with st.spinner("Predicting..."):
                # Simulate a delay for the animation effect
                time.sleep(2)

                new_data_point = pd.DataFrame([[age, income, score]], columns=required_columns)

                # Normalize the new data point
                new_data_normalized = scaler.transform(new_data_point)

                # Perform clustering
                closest_cluster = kmeans.predict(new_data_normalized)[0]

                # Display the result
                st.success(f"Predicted Cluster: {closest_cluster}")
                st.write(cluster_specs[closest_cluster])
    else:
        st.error("The uploaded file does not contain the required columns.")
