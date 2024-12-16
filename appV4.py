import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
import os

# Ignore warnings
warnings.filterwarnings('ignore')

# Enhanced Custom CSS
st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 20px;
            background-color: #f4f4f4;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .cluster-card {
            padding: 15px;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 10px;
            margin: 10px 0;
        }
        .cluster-badge {
            display: inline-block;
            padding: 5px 10px;
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            font-weight: bold;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #f0f2f6;
            color: #333;
            font-weight: bold;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

class DBSCANCustomerSegmentation:
    def __init__(self, csv_path='Customers Dataset DBSCAN.csv'):
        try:
            # Load the dataset with error handling
            if not os.path.exists(csv_path):
                st.error(f"Error: File {csv_path} not found.")
                raise FileNotFoundError(f"Could not find {csv_path}")

            # Try multiple encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(csv_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                st.error("Could not read the CSV file with any standard encoding")
                raise

            # Validate required columns
            required_columns = ['Income (INR)', 'Spending  (1-100)', 'Age', 'Gender']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Perform DBSCAN clustering
            self.perform_clustering()

        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}")
            raise

    def perform_clustering(self):
        # Select features for clustering
        features = ['Income (INR)', 'Spending  (1-100)']
        X = self.df[features].dropna()

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform DBSCAN clustering
        db = DBSCAN(eps=0.5, min_samples=10).fit(X_scaled)
        self.df.loc[X.index, 'Cluster'] = db.labels_

        # Fill NaN clusters with a separate label
        self.df['Cluster'] = self.df['Cluster'].fillna(-2)
        
        # Analyze cluster characteristics
        self.analyze_clusters()

    def analyze_clusters(self):
        # Group by clusters and compute statistics
        self.cluster_summary = self.df[self.df['Cluster'] != -2].groupby('Cluster').agg({
            'Cluster': 'size',
            'Income (INR)': ['mean', 'median'],
            'Spending  (1-100)': ['mean', 'median'],
            'Age': ['mean', 'min', 'max'],
            'Gender': lambda x: x.value_counts().to_dict()
        })
        
        # Rename columns for clarity
        self.cluster_summary.columns = [
            'Count', 
            'Mean_Income', 'Median_Income', 
            'Mean_Spending', 'Median_Spending',
            'Mean_Age', 'Min_Age', 'Max_Age', 
            'Gender_Distribution'
        ]

        # Add cluster descriptions
        self.cluster_descriptions = {
            -1: "🚨 Noise Points (Outliers)",
            0: "💼 Mainstream Customers",
            1: "📉 Low Spending Segment",
            2: "💎 High Spending Premium Segment"
        }

    def create_visualizations(self):
        plt.figure(figsize=(12, 8))
        
        # Color map for clusters
        colors = ['gray', 'blue', 'green', 'red', 'purple', 'orange']
        
        # Scatter plot of Income vs Spending by Cluster
        for i, cluster in enumerate(sorted(self.df['Cluster'].unique())):
            if cluster != -2:  # Exclude rows without cluster assignment
                cluster_data = self.df[self.df['Cluster'] == cluster]
                plt.scatter(
                    cluster_data['Income (INR)'], 
                    cluster_data['Spending  (1-100)'], 
                    label=f'Cluster {cluster}: {self.cluster_descriptions.get(cluster, "Undefined")}',
                    color=colors[i % len(colors)],
                    alpha=0.7
                )
        
        plt.title('DBSCAN Clustering: Income vs Spending')
        plt.xlabel('Income (INR)')
        plt.ylabel('Spending Score')
        plt.legend()
        plt.grid(True)
        
        return plt

def main():
    # Set page title and icon
    st.set_page_config(page_title="DBSCAN Customer Segmentation", page_icon="📊")
    
    # Header
    st.markdown('<div class="main-header"><h1>DBSCAN Customer Segmentation</h1></div>', unsafe_allow_html=True)
    
    try:
        # Initialize the model
        model = DBSCANCustomerSegmentation()
    except Exception as e:
        st.error("Failed to initialize the model. Please check your dataset.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Cluster Analysis", "📊 Cluster Overview", "📈 Visualizations", "📋 Full Dataset"])
    
    with tab1:
        # Sidebar for parameter tuning
        with st.sidebar:
            st.markdown("### 🛠️ DBSCAN Parameters")
            eps = st.slider("Epsilon (neighborhood distance)", 0.1, 2.0, 0.5)
            min_samples = st.slider("Minimum Samples", 5, 20, 10)
            
            if st.button("Re-run Clustering"):
                # Perform clustering with new parameters
                try:
                    features = ['Income (INR)', 'Spending  (1-100)']
                    X = model.df[features].dropna()
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
                    model.df.loc[X.index, 'Cluster'] = db.labels_
                    model.df['Cluster'] = model.df['Cluster'].fillna(-2)
                    
                    model.analyze_clusters()
                    st.success("Clustering re-run successfully!")
                except Exception as e:
                    st.error(f"Error in re-clustering: {e}")
        
        # Cluster Details
        st.markdown("### 📊 Cluster Characteristics")
        for cluster, description in model.cluster_descriptions.items():
            if cluster in model.cluster_summary.index:
                cluster_info = model.cluster_summary.loc[cluster]
                with st.expander(f"Cluster {cluster} | {description}"):
                    st.markdown(f"""
                        <div class="cluster-card">
                            <p>👥 Count: {cluster_info['Count']} customers</p>
                            <p>💰 Mean Income: ₹{cluster_info['Mean_Income']:,.2f}</p>
                            <p>🛍️ Mean Spending Score: {cluster_info['Mean_Spending']:.2f}</p>
                            <p>📅 Age Range: {cluster_info['Min_Age']:.0f} - {cluster_info['Max_Age']:.0f} years</p>
                            <p>👤 Gender Distribution: {cluster_info['Gender_Distribution']}</p>
                        </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 Detailed Cluster Summary")
        st.dataframe(model.cluster_summary)
    
    with tab3:
        st.markdown("### 📈 Cluster Visualization")
        visualization = model.create_visualizations()
        st.pyplot(visualization)
    
    with tab4:
        st.markdown("### 📋 Full Dataset")
        st.dataframe(model.df.style.background_gradient(subset=['Income (INR)', 'Spending  (1-100)']))

if __name__ == '__main__':
    main()
