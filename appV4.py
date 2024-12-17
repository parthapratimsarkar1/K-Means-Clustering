import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import os

class CustomerSegmentation:
    def __init__(self, csv_path='Customers Dataset DBSCAN.csv'):
        """
        Initialize customer segmentation with DBSCAN clustering
        
        Parameters:
        -----------
        csv_path : str, optional (default='Customers Dataset DBSCAN.csv')
            Path to the input CSV file containing customer data
        """
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
        Load customer data from CSV file
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file
        """
        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        for encoding in encodings:
            try:
                self.df = pd.read_csv(csv_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not read CSV file with any standard encoding")
        
        # Validate required columns
        required_columns = ['Age', 'Income (INR)', 'Spending (1-100)', 'Gender']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        print(f"Data loaded successfully. Shape: {self.df.shape}")
    
    def preprocess(self):
        """
        Preprocess data for clustering:
        1. Select features
        2. Scale features
        """
        # Select features for clustering
        self.features = ['Age', 'Income (INR)', 'Spending (1-100)']
        
        # Create scaler
        self.scaler = StandardScaler()
        
        # Scale features
        self.scaled_features = self.scaler.fit_transform(self.df[self.features])
        
        print("Data preprocessed and scaled.")
    
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
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.df['Cluster'] = dbscan.fit_predict(self.scaled_features)
        
        # Count number of clusters (excluding noise points)
        unique_clusters = len(set(self.df['Cluster'])) - (1 if -1 in self.df['Cluster'] else 0)
        print(f"DBSCAN clustering complete. Found {unique_clusters} clusters.")
    
    def analyze_clusters(self):
        """
        Analyze characteristics of each cluster
        """
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
        for cluster, stats in self.cluster_stats.items():
            print(f"\nCluster {cluster} Statistics:")
            for key, value in stats.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    def visualize_clusters(self):
        """
        Create visualizations of the clusters
        """
        # Set up the plots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Segmentation Cluster Analysis', fontsize=16)
        
        # 1. Cluster Distribution
        cluster_sizes = self.df['Cluster'].value_counts()
        axs[0, 0].pie(cluster_sizes, labels=[f'Cluster {c}' for c in cluster_sizes.index], 
                      autopct='%1.1f%%')
        axs[0, 0].set_title('Cluster Distribution')
        
        # 2. Income vs Spending Scatter Plot
        scatter = axs[0, 1].scatter(
            self.df['Income (INR)'], 
            self.df['Spending (1-100)'], 
            c=self.df['Cluster'], 
            cmap='viridis'
        )
        axs[0, 1].set_title('Income vs Spending by Cluster')
        axs[0, 1].set_xlabel('Income (INR)')
        axs[0, 1].set_ylabel('Spending Score')
        plt.colorbar(scatter, ax=axs[0, 1], label='Cluster')
        
        # 3. Box Plot of Age by Cluster
        sns.boxplot(x='Cluster', y='Age', data=self.df, ax=axs[1, 0])
        axs[1, 0].set_title('Age Distribution by Cluster')
        
        # 4. Heatmap of Cluster Characteristics
        cluster_summary = self.df.groupby('Cluster')[self.features].mean()
        sns.heatmap(cluster_summary, annot=True, cmap='coolwarm', ax=axs[1, 1])
        axs[1, 1].set_title('Cluster Characteristics Heatmap')
        
        plt.tight_layout()
        plt.show()
    
    def export_clustered_data(self, output_path='clustered_customers.csv'):
        """
        Export the clustered data to a new CSV file
        
        Parameters:
        -----------
        output_path : str, optional (default='clustered_customers.csv')
            Path to save the clustered dataset
        """
        self.df.to_csv(output_path, index=False)
        print(f"Clustered data exported to {output_path}")

def main():
    # Initialize customer segmentation
    segmentation = CustomerSegmentation()
    
    # Visualize clusters
    segmentation.visualize_clusters()
    
    # Export clustered data
    segmentation.export_clustered_data()

if __name__ == '__main__':
    main()
