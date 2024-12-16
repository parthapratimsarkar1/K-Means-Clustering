import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="DBSCAN Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

class DBSCANCustomerSegmentation:
    def __init__(self, csv_path='Customers Dataset DBSCAN.csv'):
        try:
            # Load the dataset
            self.df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_columns = ['Income (INR)', 'Spending  (1-100)']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
                st.error(f"Columns in dataset: {list(self.df.columns)}")
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Prepare data for clustering
            self.prepare_data()
            
            # Perform clustering
            self.perform_clustering()
            
            # Analyze clusters
            self.analyze_clusters()
        
        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}")
            raise
    
    def prepare_data(self):
        # Select features for clustering
        self.features = ['Income (INR)', 'Spending  (1-100)']
        
        # Remove any rows with missing values
        self.X = self.df[self.features].dropna()
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
    
    def perform_clustering(self):
        # Perform DBSCAN clustering
        self.dbscan = DBSCAN(eps=0.5, min_samples=10)
        self.labels = self.dbscan.fit_predict(self.X_scaled)
        
        # Add cluster labels to the dataframe
        self.X['Cluster'] = self.labels
    
    def analyze_clusters(self):
        # Analyze cluster characteristics
        self.cluster_summary = self.X.groupby('Cluster').agg(
            Count=('Cluster', 'size'),
            Mean_Income=('Income (INR)', 'mean'),
            Median_Income=('Income (INR)', 'median'),
            Mean_Spending=('Spending  (1-100)', 'mean'),
            Median_Spending=('Spending  (1-100)', 'median')
        )
        
        # Create cluster descriptions matching the exact clusters in your data
        self.cluster_descriptions = {
            -1: "🔍 Noise Points (Outliers)",
            0: "💼 Standard Customers",
            1: "💰 Low Spending Group",
            2: "🌟 High Spending Group",
            3: "🔒 Constrained Spending Group"
        }
    
    def create_visualizations(self):
        # Create scatter plot of clusters
        plt.figure(figsize=(12, 8))
        
        # Plot noise points (Cluster -1) separately
        noise_data = self.X[self.X['Cluster'] == -1]
        plt.scatter(noise_data['Income (INR)'], 
                    noise_data['Spending  (1-100)'], 
                    color='red', label='Noise Points', marker='x')
        
        # Plot other clusters
        for cluster_label in sorted(self.X['Cluster'].unique()):
            if cluster_label == -1:
                continue
            cluster_data = self.X[self.X['Cluster'] == cluster_label]
            plt.scatter(cluster_data['Income (INR)'], 
                        cluster_data['Spending  (1-100)'], 
                        label=f'Cluster {cluster_label}')
        
        plt.title('DBSCAN Clustering: Income vs Spending Score')
        plt.xlabel('Income (INR)')
        plt.ylabel('Spending Score')
        plt.legend()
        plt.grid(True)
        
        return plt

def main():
    # Header
    st.markdown('<div class="main-header"><h1>DBSCAN Customer Segmentation</h1></div>', unsafe_allow_html=True)
    
    try:
        # Initialize the segmentation model
        model = DBSCANCustomerSegmentation()
    except Exception as e:
        st.error("Failed to initialize the model. Please check your dataset.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Cluster Overview", "📈 Visualizations", "🔍 Detailed Analysis", "📋 Full Dataset"])
    
    with tab1:
        st.markdown("### 📊 Cluster Composition")
        
        # Display cluster summary
        st.dataframe(model.cluster_summary)
        
        # Detailed cluster description
        st.markdown("### 🏷️ Cluster Descriptions")
        for cluster, description in model.cluster_descriptions.items():
            with st.expander(f"Cluster {cluster} | {description}"):
                if cluster in model.cluster_summary.index:
                    cluster_info = model.cluster_summary.loc[cluster]
                    st.markdown(f"""
                        <div class="cluster-card">
                            <p>👥 Number of Customers: {cluster_info['Count']}</p>
                            <p>💰 Average Income: ₹{cluster_info['Mean_Income']:,.2f}</p>
                            <p>🛍️ Average Spending Score: {cluster_info['Mean_Spending']:.2f}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.write("No data for this cluster")
    
    with tab2:
        st.markdown("### 📈 Cluster Visualization")
        
        # Create and display the visualization
        visualization = model.create_visualizations()
        st.pyplot(visualization)
        
        # Additional insights
        st.markdown("### 🔍 Clustering Insights")
        st.write("Red 'x' points represent noise points or outliers detected by DBSCAN.")
    
    with tab3:
        st.markdown("### 🔬 Detailed Cluster Analysis")
        
        # Select cluster for detailed view
        selected_cluster = st.selectbox("Select Cluster", 
                                        sorted(model.X['Cluster'].unique()))
        
        # Filter data for selected cluster
        cluster_data = model.X[model.X['Cluster'] == selected_cluster]
        
        # Display detailed statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Descriptive Statistics")
            st.dataframe(cluster_data.describe())
        
        with col2:
            st.markdown("#### 📈 Distribution")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Income distribution
            sns.histplot(cluster_data['Income (INR)'], kde=True, ax=ax1)
            ax1.set_title(f'Income Distribution - Cluster {selected_cluster}')
            
            # Spending distribution
            sns.histplot(cluster_data['Spending  (1-100)'], kde=True, ax=ax2)
            ax2.set_title(f'Spending Distribution - Cluster {selected_cluster}')
            
            st.pyplot(fig)
    
    with tab4:
        st.markdown("### 📋 Full Dataset")
        
        # Display full dataset with styling
        st.dataframe(
            model.X.style.background_gradient(subset=['Income (INR)', 'Spending  (1-100)'])
        )
        
        # Additional dataset information
        st.markdown("#### 📊 Dataset Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Records", len(model.X))
            st.metric("Unique Clusters", len(model.X['Cluster'].unique()))
        
        with col2:
            st.metric("Avg Income", f"₹{model.X['Income (INR)'].mean():,.2f}")
            st.metric("Avg Spending Score", f"{model.X['Spending  (1-100)'].mean():.2f}")

if __name__ == '__main__':
    main()