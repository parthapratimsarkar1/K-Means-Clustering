import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Set page configuration
st.set_page_config(
    page_title="DBSCAN Customer Segmentation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        .metric-container p {
            margin: 5px 0;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

class DBSCANCustomerSegmentation:
    def __init__(self, csv_path='Customers Dataset DBSCAN.csv'):
        # Load the dataset
        self.df = pd.read_csv(csv_path)

        # Predefined cluster descriptions
        self.cluster_descriptions = {
            0: {
                'name': 'Conservative Spenders',
                'description': 'High Income, Moderate Spending',
                'color': '#3498db',
                'icon': '💰'
            },
            1: {
                'name': 'Balanced Customers',
                'description': 'Moderate Income, Balanced Spending',
                'color': '#2ecc71',
                'icon': '⚖️'
            },
            2: {
                'name': 'Premium Customers',
                'description': 'High Income, High Spending',
                'color': '#9b59b6',
                'icon': '💎'
            },
            3: {
                'name': 'Risk Group',
                'description': 'High Income, Low Spending',
                'color': '#e74c3c',
                'icon': '⚠️'
            },
            -1: {
                'name': 'Unique Outliers',
                'description': 'Non-Standard Customer Profile',
                'color': '#f39c12',
                'icon': '🌟'
            }
        }

        # Prepare data for clustering
        self.prepare_data()

    def prepare_data(self):
        # Select features for clustering
        features = ['Income (INR)', 'Spending (1-100)']

        # Scale the features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(self.df[features])

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.df['Cluster'] = dbscan.fit_predict(scaled_features)

        # Compute cluster centroids
        self.compute_cluster_centroids(scaled_features)

    def compute_cluster_centroids(self, scaled_features):
        self.cluster_centroids = {}
        for cluster in self.df['Cluster'].unique():
            cluster_points = scaled_features[self.df['Cluster'] == cluster]
            centroid = cluster_points.mean(axis=0)
            self.cluster_centroids[cluster] = centroid

    def predict_customer_cluster(self, income, spending):
        input_data = self.scaler.transform([[income, spending]])
        distances = {}
        for cluster, centroid in self.cluster_centroids.items():
            distance = np.linalg.norm(input_data - centroid)
            distances[cluster] = distance
        predicted_cluster = min(distances, key=distances.get)
        return predicted_cluster

    def analyze_customer_profile(self, customer_id, gender, age, income, spending):
        cluster = self.predict_customer_cluster(income, spending)
        return {
            'customer_id': customer_id,
            'cluster': cluster,
            'cluster_details': self.cluster_descriptions[cluster],
            'risk_score': self._calculate_risk_score(cluster, income, spending, age),
            'spending_potential': self._calculate_spending_potential(cluster, income, spending),
            'personalized_insights': self._generate_personalized_insights(cluster, gender, age)
        }

    def _calculate_risk_score(self, cluster, income, spending, age):
        # Implement a more sophisticated risk scoring model
        # Consider factors like age, income, spending, and cluster-specific risks
        pass

    def _calculate_spending_potential(self, cluster, income, spending):
        # Implement a more accurate spending potential estimation model
        # Consider factors like income, spending habits, and cluster-specific potential
        pass

    def _generate_personalized_insights(self, cluster, gender, age):
        # Implement a more advanced personalization engine
        # Consider using NLP techniques to generate tailored insights
        pass

    def create_visualizations(self):
        # Cluster Distribution
        plt.figure(figsize=(10, 6))
        cluster_counts = self.df['Cluster'].value_counts()
        cluster_counts.plot(kind='bar')
        plt.title('Customer Cluster Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Customers')
        cluster_dist_fig = plt.gcf()
        plt.close()

        # Income vs Spending Scatter
        plt.figure(figsize=(10, 6))
        for cluster in self.df['Cluster'].unique():
            cluster_data = self.df[self.df['Cluster'] == cluster]
            plt.scatter(cluster_data['Income (INR)'], cluster_data['Spending (1-100)'],
                        label=f'Cluster {cluster}', alpha=0.7)
        plt.title('Income vs Spending by Cluster')
        plt.xlabel('Income (INR)')
        plt.ylabel('Spending Score')
        plt.legend()
        income_spending_fig = plt.gcf()
        plt.close()

        return {
            'cluster_distribution': cluster_dist_fig,
            'income_vs_spending': income_spending_fig
        }

def main():
    # Main title
    st.markdown('<div class="main-header"><h1>DBSCAN Customer Segmentation</h1></div>', unsafe_allow_html=True)

    # Initialize model
    model = DBSCANCustomerSegmentation()

    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Customer Analysis",
        "📊 Cluster Overview",
        "📈 Visualizations",
        "📋 Full Dataset"
    ])

    # ... (rest of the code)

if __name__ == '__main__':
    main()
