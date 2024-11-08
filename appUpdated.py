import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Must be the first Streamlit command
st.set_page_config(page_title="Customer Segmentation System", layout="wide")

# CSS to add logo as a background watermark
st.markdown(
    """
    <style>
    .watermark {
        position: fixed;
        bottom: 20px;
        right: 20px;
        opacity: 0.1;
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display watermark logo
st.markdown(
    """
    <div class="watermark">
        <img src="path/to/logo.png" width="200">
    </div>
    """,
    unsafe_allow_html=True
)

# Your main Streamlit code for the app
class CustomerSegmentation:
    def __init__(self, csv_path='Customers_Segmentation_with_Clusters.csv'):
        """Initialize the Customer Segmentation model"""
        try:
            self.df = pd.read_csv(csv_path)
            self.process_data()
            self.analyze_clusters()
            self.cluster_descriptions = {
                0: "Conservative Spenders (High Income): Customers earning high but spending less",
                1: "Balanced Customers: Average in terms of earning and spending",
                2: "Risk Customers: Earning Low and Spending high",
                3: "Risk Group: Earning less but spending more",
                4: "Budget Conscious: Earning less, spending less",
                5: "Moderate Savers: Average earning, spending less"
            }
        except FileNotFoundError:
            st.error(f"Error: Could not find {csv_path}")
    
    # (The rest of your code remains unchanged)
    
if __name__ == '__main__':
    main()
