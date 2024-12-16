import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with cluster number styling
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

class CustomerSegmentation:
    def __init__(self, csv_path='Customer-Dataset-With-Clustered.csv'):
        try:
            # Add error handling for file loading
            if not os.path.exists(csv_path):
                st.error(f"Error: File {csv_path} not found. Please check the file path.")
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
            required_columns = ['Age', 'Income (INR)', 'Spending (1-100)', 'Gender', 'Cluster']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
                st.error(f"Columns in dataset: {list(self.df.columns)}")
                raise ValueError(f"Missing required columns: {missing_columns}")

            self.process_data()
            self.analyze_clusters()
            self.generate_cluster_descriptions()

        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}")
            raise
            
    def process_data(self):
        le = LabelEncoder()
        self.df['Gender_Encoded'] = le.fit_transform(self.df['Gender'])
        self.gender_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        
        self.features = ['Age', 'Income (INR)', 'Spending (1-100)']
        self.scaler = StandardScaler()
        
        try:
            self.scaled_features = self.scaler.fit_transform(self.df[self.features])
        except Exception as e:
            st.error(f"Error in data scaling: {e}")
            raise
    
    def generate_cluster_descriptions(self):
        self.cluster_descriptions = {
            0: "🏦 Conservative Spenders (High Income)",
            1: "⚖️ Balanced Customers",
            2: "💎 Premium Customers",
            3: "⚠️ Risk Group"
        }
        
        self.cluster_details = {
            0: "High income earners with conservative spending habits",
            1: "Customers with balanced earning and spending patterns",
            2: "High-income customers with premium spending habits",
            3: "Lower income group with higher spending patterns"
        }
    
    def analyze_clusters(self):
        unique_clusters = sorted(self.df['Cluster'].unique())
        self.cluster_info = {}
        
        for cluster in unique_clusters:
            cluster_data = self.df[self.df['Cluster'] == cluster]
            
            if len(cluster_data) > 0:
                self.cluster_info[cluster] = {
                    'size': len(cluster_data),
                    'avg_age': round(cluster_data['Age'].mean(), 1),
                    'avg_income': round(cluster_data['Income (INR)'].mean(), 1),
                    'avg_spending': round(cluster_data['Spending (1-100)'].mean(), 1),
                    'gender_distribution': cluster_data['Gender'].value_counts().to_dict()
                }
            else:
                self.cluster_info[cluster] = {
                    'size': 0,
                    'avg_age': 0,
                    'avg_income': 0,
                    'avg_spending': 0,
                    'gender_distribution': {}
                }

    def predict_segment(self, customer_id, gender, age, income, spending):
        input_data = np.array([[age, income, spending]])
        scaled_input = self.scaler.transform(input_data)
        
        distances = []
        for cluster in self.df['Cluster'].unique():
            cluster_data = self.df[self.df['Cluster'] == cluster]
            cluster_center = cluster_data[self.features].mean()
            scaled_center = self.scaler.transform([cluster_center])
            distance = np.linalg.norm(scaled_input - scaled_center)
            distances.append((cluster, distance))
        
        predicted_cluster = min(distances, key=lambda x: x[1])[0]
        return predicted_cluster, self.cluster_info[predicted_cluster]

def main():
    # Header with professional styling
    st.markdown('<div class="main-header"><h1>Customer Segmentation System</h1></div>', unsafe_allow_html=True)
    
    try:
        model = CustomerSegmentation()
    except Exception as e:
        st.error("Failed to initialize the model. Please check your dataset.")
        return
    
    # Sidebar with customer inputs
    with st.sidebar:
        st.markdown("### 📊 Customer Profile Analysis")
        st.markdown("---")
        
        if st.button("View Full Dataset"):
            st.dataframe(model.df.style.background_gradient(subset=['Income (INR)', 'Spending (1-100)'])
                         .format({'Cluster': 'Cluster {}'}))
        
        st.markdown("---")
        
        customer_id = st.text_input("📋 Customer ID")
        gender = st.selectbox("👤 Gender", options=["Male", "Female"])
        age = st.number_input("🎂 Age", min_value=0, max_value=100, value=30)
        income = st.number_input("💵 Income (INR)", min_value=0, value=50000)
        spending = st.number_input("🛍️ Spending (1-100)", min_value=0, max_value=100, value=50)
        
        if st.button("Analyze Customer"):
            if all([customer_id, gender, age, income, spending]):
                try:
                    cluster, info = model.predict_segment(customer_id, gender, age, income, spending)
                    st.markdown(f"""
                        <div class="cluster-card">
                            <span class="cluster-badge">Cluster {cluster}</span>
                            <h4>{model.cluster_descriptions[cluster]}</h4>
                            <p>{model.cluster_details[cluster]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error in customer analysis: {e}")
            else:
                st.warning("⚠️ Please complete all fields")
    
    # Cluster Overview
    st.markdown("### 📊 Cluster Overview")
    for cluster in sorted(model.cluster_descriptions.keys()):
        with st.expander(f"Cluster {cluster} | {model.cluster_descriptions[cluster]}"):
            info = model.cluster_info.get(cluster, {})
            st.markdown(f"""
                <div class="cluster-card">
                    <p>{model.cluster_details[cluster]}</p>
                    <p>👥 Cluster Size: {info['size']} customers</p>
                    <p>📅 Average Age: {info['avg_age']} years</p>
                    <p>💰 Average Income: ₹{info['avg_income']:,}</p>
                    <p>🛍️ Average Spending Score: {info['avg_spending']}</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
