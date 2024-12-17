import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide", page_icon="📊")

# Add custom CSS for aesthetics
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with emoji
st.title("📊 Customer Segmentation System")

# Sidebar for user input
st.sidebar.header("📂 Upload Your Dataset")
file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if file:
    # Load the dataset
    df = pd.read_csv(file)
    st.write("### Uploaded Dataset")
    st.dataframe(df.style.background_gradient(cmap="coolwarm"))

    # Select columns for clustering
    st.sidebar.header("⚙ Clustering Configuration")
    columns = st.sidebar.multiselect("Select Features for Clustering", df.columns)

    if columns:
        X = df[columns].values

        # Preprocess data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select clustering algorithm
        algo = st.sidebar.selectbox(
            "Choose Clustering Algorithm",
            ("KMeans", "DBSCAN", "Hierarchical Clustering")
        )

        if algo == "KMeans":
            n_clusters = st.sidebar.slider("Select Number of Clusters (K)", 2, 10, 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X_scaled)

        elif algo == "DBSCAN":
            eps = st.sidebar.slider("Select Epsilon (eps)", 0.1, 5.0, 0.5, step=0.1)
            min_samples = st.sidebar.slider("Select Min Samples", 1, 10, 5)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)

        elif algo == "Hierarchical Clustering":
            n_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)
            linkage = st.sidebar.selectbox("Select Linkage Method", ["ward", "complete", "average", "single"])
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = hierarchical.fit_predict(X_scaled)

        # Add cluster labels to the dataset
        df['Cluster'] = labels
        st.write("### Clustered Dataset")
        st.dataframe(df.style.background_gradient(cmap="viridis"))

        # Visualize clusters (2D)
        st.write("### Cluster Visualization")
        if len(columns) >= 2:
            x_col = st.selectbox("Select X-axis Feature", columns, index=0)
            y_col = st.selectbox("Select Y-axis Feature", columns, index=1)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x=df[x_col], y=df[y_col], hue=labels, palette="viridis", style=labels, ax=ax
            )
            ax.set_title("Clustering Results", fontsize=16)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

        else:
            st.warning("Please select at least two features to visualize the clusters.")

        # Download clustered dataset
        st.sidebar.header("💾 Download Results")
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download Clustered Dataset",
            data=csv,
            file_name="clustered_dataset.csv",
            mime="text/csv"
        )

    else:
        st.warning("Please select features for clustering.")
else:
    st.info("Awaiting CSV file upload...")
