import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

class CustomerSegmentation:
    def __init__(self, csv_path='Customers Dataset DBSCAN.csv'):
        """
        Initialize the customer segmentation analysis
        
        Args:
            csv_path (str): Path to the customer dataset
        """
        # Load the dataset
        try:
            self.data = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: File {csv_path} not found.")
            return
        
        # Predefined cluster descriptions
        self.cluster_descriptions = {
            -1: {
                'name': 'Outliers',
                'description': 'Unique Customer Profiles',
                'color': '#FFA500'
            },
            0: {
                'name': 'Moderate Customers',
                'description': 'Average Income and Spending',
                'color': '#3498db'
            },
            1: {
                'name': 'Low Spending Customers',
                'description': 'Low Income, Conservative Spending',
                'color': '#95a5a6'
            },
            2: {
                'name': 'High Spending Customers',
                'description': 'High Income, High Spending',
                'color': '#2ecc71'
            },
            3: {
                'name': 'High Income, Low Spending',
                'description': 'High Potential Customers',
                'color': '#e74c3c'
            }
        }
        
        # Prepare and cluster data
        self.prepare_data()
    
    def prepare_data(self):
        """
        Prepare data for clustering by selecting and scaling features
        """
        # Select relevant features (correcting potential column name issues)
        features = ['Income (INR)', 'Spending  (1-100)']
        
        # Drop rows with missing values
        X = self.data[features].dropna()
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform DBSCAN clustering
        db = DBSCAN(eps=0.5, min_samples=10)
        labels = db.fit_predict(X_scaled)
        
        # Add cluster labels to the dataset
        X['Cluster'] = labels
        self.clustered_data = X
    
    def analyze_clusters(self):
        """
        Analyze and print cluster characteristics
        
        Returns:
            pd.DataFrame: Cluster summary statistics
        """
        cluster_summary = self.clustered_data.groupby('Cluster').agg(
            Count=('Cluster', 'size'),
            Mean_Income=('Income (INR)', 'mean'),
            Median_Income=('Income (INR)', 'median'),
            Mean_Spending=('Spending  (1-100)', 'mean'),
            Median_Spending=('Spending  (1-100)', 'median')
        )
        
        print("Cluster Characteristics Summary:\n")
        print(cluster_summary)
        return cluster_summary
    
    def visualize_clusters(self):
        """
        Create visualizations of the clusters
        """
        plt.figure(figsize=(12, 8))
        
        # Scatter plot of clusters
        for cluster_label in sorted(self.clustered_data['Cluster'].unique()):
            cluster_data = self.clustered_data[self.clustered_data['Cluster'] == cluster_label]
            plt.scatter(
                cluster_data['Income (INR)'], 
                cluster_data['Spending  (1-100)'], 
                label=f'Cluster {cluster_label}: {self.cluster_descriptions[cluster_label]["name"]}',
                color=self.cluster_descriptions[cluster_label]['color'],
                alpha=0.7
            )
        
        plt.title('Customer Segmentation: Income vs Spending', fontsize=15)
        plt.xlabel('Income (INR)', fontsize=12)
        plt.ylabel('Spending Score (1-100)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def detailed_cluster_analysis(self):
        """
        Provide detailed insights for each cluster
        """
        for cluster, details in self.cluster_descriptions.items():
            if cluster in self.clustered_data['Cluster'].unique():
                cluster_data = self.clustered_data[self.clustered_data['Cluster'] == cluster]
                
                print(f"\n--- Cluster {cluster}: {details['name']} ---")
                print(f"Description: {details['description']}")
                print(f"Number of Customers: {len(cluster_data)}")
                print(f"Average Income: ₹{cluster_data['Income (INR)'].mean():,.2f}")
                print(f"Average Spending Score: {cluster_data['Spending  (1-100)'].mean():.2f}")
    
    def export_cluster_data(self, output_path='clustered_customers.csv'):
        """
        Export clustered data to a CSV file
        
        Args:
            output_path (str): Path to save the clustered data
        """
        # Merge original data with cluster labels
        export_data = self.data.copy()
        export_data['Cluster'] = self.clustered_data['Cluster']
        export_data.to_csv(output_path, index=False)
        print(f"Clustered data exported to {output_path}")

def main():
    # Initialize the customer segmentation analysis
    segmentation = CustomerSegmentation()
    
    # Analyze clusters
    segmentation.analyze_clusters()
    
    # Visualize clusters
    segmentation.visualize_clusters()
    
    # Detailed cluster analysis
    segmentation.detailed_cluster_analysis()
    
    # Export clustered data
    segmentation.export_cluster_data()

if __name__ == "__main__":
    main()
