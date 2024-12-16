import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(
    page_title="DBSCAN Customer Segmentation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for styling
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
    </style>
""", unsafe_allow_html=True)

class DBCustomerSegmentation:
    def __init__(self, csv_path='Customers Dataset DBSCAN.csv'):
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                st.error(f"Error: File {csv_path} not found. Please check the file path.")
                raise FileNotFoundError(f"Could not find {csv_path}")
            
            self.df = pd.read_csv(csv_path)

            # Validate required columns
            required_columns = ['CustomerID', 'Name', 'Gender', 'Age', 'Income (INR)', 'Spending (1-100)', 'CIBIL Score']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")

            self.process_data()
            self.cluster_data()

        except Exception as e:
            st.error(f"An error occurred: {e}")
            raise

    def process_data(self):
        # Select features for clustering
        self.features = ['Age', 'Income (INR)', 'Spending (1-100)', 'CIBIL Score']
        
        # Standardizing the features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[self.features])

    def cluster_data(self):
        # Apply DBSCAN for clustering
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.df['Cluster'] = self.dbscan.fit_predict(self.scaled_features)
        
        # Identify noise points (label = -1)
        self.noise_points = len(self.df[self.df['Cluster'] == -1])

    def create_visualizations(self):
        # Create visualizations for DBSCAN
        plt.figure(figsize=(10, 6))

        # Plot clusters
        plt.scatter(self.df['Income (INR)'], self.df['Spending (1-100)'], c=self.df['Cluster'], cmap='viridis', s=50)
        plt.title('DBSCAN Clustering: Income vs Spending')
        plt.xlabel('Income (INR)')
        plt.ylabel('Spending (1-100)')
        plt.colorbar(label='Cluster')
        st.pyplot()

    def cluster_summary(self):
        cluster_info = {}
        
        # Summary statistics for each cluster
        for cluster in np.unique(self.df['Cluster']):
            cluster_data = self.df[self.df['Cluster'] == cluster]
            cluster_info[cluster] = {
                'size': len(cluster_data),
                'avg_age': round(cluster_data['Age'].mean(), 1),
                'avg_income': round(cluster_data['Income (INR)'].mean(), 1),
                'avg_spending': round(cluster_data['Spending (1-100)'].mean(), 1),
                'avg_cibil': round(cluster_data['CIBIL Score'].mean(), 1)
            }
        
        return cluster_info

    def noise_summary(self):
        return self.noise_points

def main():
    # Header
    st.markdown('<div class="main-header"><h1>DBSCAN Customer Segmentation</h1></div>', unsafe_allow_html=True)
    
    try:
        model = DBCustomerSegmentation()
    except Exception as e:
        st.error(f"Failed to initialize the model. Please check your dataset.")
        return
    
    # Main tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Customer Analysis", "📊 Data Overview", "📈 Visualizations", "📋 Full Dataset"])
    
    with tab1:
        st.markdown("### 📊 Customer Segment Analysis")
        
        # Analyze Noise Points and Silhouette Coefficient
        st.write(f"Total Noise Points: {model.noise_summary()}")
        
        # Show summary of clusters
        cluster_info = model.cluster_summary()
        for cluster, info in cluster_info.items():
            st.markdown(f"""
                <div class="cluster-card">
                    <span class="cluster-badge">Cluster {cluster}</span>
                    <p>👥 Cluster Size: {info['size']} customers</p>
                    <p>📅 Average Age: {info['avg_age']} years</p>
                    <p>💰 Average Income: ₹{info['avg_income']}</p>
                    <p>🛍️ Average Spending: {info['avg_spending']}</p>
                    <p>📊 Average CIBIL Score: {info['avg_cibil']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 Comprehensive Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔢 Basic Statistics")
            st.dataframe(model.df.describe().T.style.background_gradient())
        
        with col2:
            st.markdown("#### 📊 Cluster Composition")
            cluster_composition = model.df['Cluster'].value_counts()
            st.dataframe(cluster_composition)
        
        st.markdown("#### 🔍 Detailed Cluster Breakdown")
        st.dataframe(model.df.groupby('Cluster')[['Age', 'Income (INR)', 'Spending (1-100)', 'CIBIL Score']].agg(['mean', 'median', 'std']).style.background_gradient())

    with tab3:
        st.markdown("### 📈 Visualizations")
        model.create_visualizations()

    with tab4:
        st.markdown("### 📋 Full Dataset")
        st.dataframe(model.df.style.background_gradient(subset=['Income (INR)', 'Spending (1-100)', 'CIBIL Score']))

if __name__ == '__main__':
    main()
