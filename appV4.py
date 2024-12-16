import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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

        self.prepare_data()

    def prepare_data(self):
        # Print column names for debugging
        print(self.df.columns)

        # Select the relevant features (correct for extra spaces)
        features = ['Income (INR)', 'Spending (1-100)']
        X = self.df[features].dropna()

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform DBSCAN clustering
        db = DBSCAN(eps=0.5, min_samples=10).fit(X_scaled)
        labels = db.labels_

        # Add the cluster labels to the original dataset
        self.df['Cluster'] = labels

    def analyze_customer_profile(self, customer_id, gender, age, income, spending):
        # ... (your customer analysis logic)
        pass

    def create_visualizations(self):
        # ... (your visualization code)
        pass

def main():
    # Main title
    st.markdown('<div class="main-header"><h1>DBSCAN Customer Segmentation</h1></div>', unsafe_allow_html=True)

    # Initialize model
    model = DBSCANCustomerSegmentation()

    # ... (rest of your Streamlit app code)

if __name__ == '__main__':
    main()
