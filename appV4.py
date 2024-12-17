import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st

class CustomerSegmentation:
    def __init__(self, csv_path='Customers Dataset DBSCAN.csv'):
        """
        Initialize customer segmentation with DBSCAN clustering
        
        Parameters:
        -----------
        csv_path : str, optional (default='Customers Dataset DBSCAN.csv')
            Path to the input CSV file containing customer data
        """
        # Define required columns explicitly
        self.required_columns = ['Age', 'Income (INR)', 'Spending (1-100)', 'Gender']
        
        # Load the data
        self.load_data(csv_path)
        
        # Preprocess the data
        self.preprocess()
        
        # Perform DBSCAN clustering
        self.apply_dbscan()
        
        # Analyze clusters
        self.analyze_clusters()
    
    def load_data(self, csv_path):
        """
        Load and validate customer data from CSV file
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file
        """
        # Extensive error checking and reporting
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                st.error(f"CSV file not found: {csv_path}")
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            # Try multiple encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            
            # Attempt to read file with different encodings
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(csv_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                st.error("Could not read CSV file with any standard encoding")
                raise ValueError("Could not read CSV file with any standard encoding")
            
            # Print available columns for debugging
            st.write("Available columns:", list(self.df.columns))
            
            # Validate required columns
            missing_columns = [col for col in self.required_columns if col not in self.df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                st.error(f"Columns in dataset: {list(self.df.columns)}")
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            st.success(f"Data loaded successfully. Shape: {self.df.shape}")
            print(f"Data loaded successfully. Shape: {self.df.shape}")
        
        except Exception as e:
            st.error(f"Error loading data: {e}")
            raise
    
    def preprocess(self):
        """
        Preprocess data for clustering:
        1. Select features
        2. Scale features
        """
        try:
            # Select features for clustering
            self.features = ['Age', 'Income (INR)', 'Spending (1-100)']
            
            # Create scaler
            self.scaler = StandardScaler()
            
            # Scale features
            self.scaled_features = self.scaler.fit_transform(self.df[self.features])
            
            st.success("Data preprocessed and scaled.")
            print("Data preprocessed and scaled.")
        
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            raise
    
    def apply_dbscan(self, eps=0.5, min_samples=5):
        """
        Apply DBSCAN clustering
        
        Parameters:
        -----------
        eps : float, optional (default=0.5)
            Maximum distance between two samples to be considered in the same neighborhood
        min_samples : int, optional (default=5)
            Minimum number of samples in a neighborhood for a point to be considered a core point
        """
        try:
            # Apply DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            self.df['Cluster'] = dbscan.fit_predict(self.scaled_features)
            
            # Count number of clusters (excluding noise points)
            unique_clusters = len(set(self.df['Cluster'])) - (1 if -1 in self.df['Cluster'] else 0)
            
            st.success(f"DBSCAN clustering complete. Found {unique_clusters} clusters.")
            print(f"DBSCAN clustering complete. Found {unique_clusters} clusters.")
        
        except Exception as e:
            st.error(f"Error in DBSCAN clustering: {e}")
            raise
    
    def analyze_clusters(self):
        """
        Analyze characteristics of each cluster
        """
        try:
            self.cluster_stats = {}
            
            for cluster in sorted(self.df['Cluster'].unique()):
                cluster_data = self.df[self.df['Cluster'] == cluster]
                
                self.cluster_stats[cluster] = {
                    'size': len(cluster_data),
                    'avg_age': cluster_data['Age'].mean(),
                    'avg_income': cluster_data['Income (INR)'].mean(),
                    'avg_spending': cluster_data['Spending (1-100)'].mean(),
                    'gender_distribution': cluster_data['Gender'].value_counts(normalize=True).to_dict()
                }
            
            # Print cluster statistics
            st.write("Cluster Statistics:")
            for cluster, stats in self.cluster_stats.items():
                st.write(f"\nCluster {cluster} Statistics:")
                for key, value in stats.items():
                    st.write(f"{key.replace('_', ' ').title()}: {value}")
        
        except Exception as e:
            st.error(f"Error in cluster analysis: {e}")
            raise

def main():
    st.title("Customer Segmentation with DBSCAN")
    
    # File uploader for CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Initialize customer segmentation with the uploaded file
            segmentation = CustomerSegmentation(os.path.join("tempDir", uploaded_file.name))
            
            # Additional Streamlit-specific visualizations could be added here
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    # Ensure temp directory exists
    os.makedirs("tempDir", exist_ok=True)
    
    main()
