import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import os

# Try to import Plotly, but prepare for fallback
try:
    import plotly.express as px
    import plotly.graph_objs as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not installed. Using Matplotlib for visualizations.")

warnings.filterwarnings('ignore')

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
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e6e9f0;
        }
        .stTabs [data-testid="stMarkdownContainer"] h3 {
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

class CustomerSegmentation:
    def __init__(self, csv_path=None):
        # Allow dynamic file selection if not provided
        if csv_path is None:
            csv_path = st.file_uploader("Upload Customer Dataset", type=['csv'])
            if csv_path is None:
                st.stop()

        try:
            # Handle both file upload and file path scenarios
            if isinstance(csv_path, str):
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
            else:
                # For uploaded file
                self.df = pd.read_csv(csv_path)

            # Standardize column names
            self.df.columns = self.df.columns.str.strip()

            # Debug information
            st.sidebar.markdown("### 📊 Dataset Information")
            st.sidebar.text(f"Total Rows: {len(self.df)}")
            st.sidebar.text(f"Total Columns: {len(self.df.columns)}")
            st.sidebar.text("Columns:")
            for col in self.df.columns:
                st.sidebar.text(f"- {col}")

            # Validate required columns with flexibility
            required_columns = {
                'Age': ['Age'],
                'Income': ['Income', 'Income (INR)', 'Annual Income'],
                'Spending': ['Spending', 'Spending Score', 'Spending (1-100)'],
                'Gender': ['Gender', 'Sex'],
                'Cluster': ['Cluster', 'Customer_Cluster', 'Segment']
            }

            # Find matching columns
            column_mapping = {}
            for req_key, possible_columns in required_columns.items():
                matched_column = next((col for col in possible_columns if col in self.df.columns), None)
                if matched_column:
                    column_mapping[req_key] = matched_column
                else:
                    st.error(f"Could not find column for {req_key}")
                    raise ValueError(f"Missing required column type: {req_key}")

            # Rename columns for consistency
            self.df.rename(columns=column_mapping, inplace=True)

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
        
        self.features = ['Age', 'Income', 'Spending']
        self.scaler = StandardScaler()
        
        try:
            self.scaled_features = self.scaler.fit_transform(self.df[self.features])
        except Exception as e:
            st.error(f"Error in data scaling: {e}")
            raise
    
    def generate_cluster_descriptions(self):
        self.cluster_descriptions = {
           -1: "🏦 Noise Points",
            0: "🏦 Conservative Spenders",
            1: "⚠️ Risk Customers",
            2: "💎 Premium Customers",
            3: "⚖️ Balanced Group"
        }
        
        self.cluster_details = {
           -1: "Outliers not fitting into main segments",
            0: "Middle Income, Moderate Spending",
            1: "Low Income, Low Spending",
            2: "High Income, High Spending",
            3: "Upper Middle Income, Low Spending"
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
                    'avg_income': round(cluster_data['Income'].mean(), 1),
                    'avg_spending': round(cluster_data['Spending'].mean(), 1),
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

    def predict_segment(self, age, income, spending):
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

    def create_visualizations(self):
        visualizations = {}

        # Age Distribution
        plt.figure(figsize=(10, 6))
        for cluster in self.df['Cluster'].unique():
            cluster_data = self.df[self.df['Cluster'] == cluster]['Age']
            plt.hist(cluster_data, alpha=0.5, label=f'Cluster {cluster}')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution Across Clusters')
        plt.legend()
        visualizations['age_distribution'] = plt

        # Income vs Spending
        plt.figure(figsize=(10, 6))
        for cluster in self.df['Cluster'].unique():
            cluster_data = self.df[self.df['Cluster'] == cluster]
            plt.scatter(cluster_data['Income'], cluster_data['Spending'], 
                        label=f'Cluster {cluster}')
        plt.xlabel('Income')
        plt.ylabel('Spending')
        plt.title('Income vs Spending by Cluster')
        plt.legend()
        visualizations['income_vs_spending'] = plt

        # Cluster Distribution
        cluster_sizes = self.df['Cluster'].value_counts()
        plt.figure(figsize=(10, 6))
        plt.pie(cluster_sizes.values, labels=[f"Cluster {x}" for x in cluster_sizes.index], autopct='%1.1f%%')
        plt.title("Customer Cluster Distribution")
        visualizations['cluster_distribution'] = plt

        return visualizations

def main():
    st.markdown('<div class="main-header"><h1>Customer Segmentation System</h1></div>', unsafe_allow_html=True)
    
    # File upload in main method as a fallback
    uploaded_file = st.file_uploader("Upload Customer Dataset", type=['csv'])
    
    if uploaded_file is not None:
        try:
            model = CustomerSegmentation(uploaded_file)
        except Exception as e:
            st.error(f"Failed to initialize the model: {e}")
            return
    else:
        st.warning("Please upload a CSV file to continue.")
        return
    
    # Main tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Customer Analysis", "📊 Data Overview", "📈 Visualizations", "📋 Full Dataset"])
    
    with tab1:
        # Sidebar with customer inputs
        with st.sidebar:
            st.markdown("### 📊 Customer Profile Analysis")
            st.markdown("---")
            
            age = st.number_input("🎂 Age", min_value=0, max_value=100, value=30)
            income = st.number_input("💵 Income", min_value=0, value=50000)
            spending = st.number_input("🛍️ Spending", min_value=0, max_value=100, value=50)
            
            if st.button("Analyze Customer"):
                try:
                    cluster, info = model.predict_segment(age, income, spending)
                    st.markdown(f"""
                        <div class="cluster-card">
                            <span class="cluster-badge">Cluster {cluster}</span>
                            <h4>{model.cluster_descriptions[cluster]}</h4>
                            <p>{model.cluster_details[cluster]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("Cluster Details:")
                    st.json(info)
                except Exception as e:
                    st.error(f"Error in customer analysis: {e}")
        
        # Cluster Overview in first tab
        st.markdown("### 📊 Cluster Overview")
        for cluster in sorted(model.cluster_descriptions.keys()):
            with st.expander(f"Cluster {cluster} | {model.cluster_descriptions[cluster]}"):
                info = model.cluster_info.get(cluster, {})
                st.markdown(f"""
                    <div class="cluster-card">
                        <p>{model.cluster_details[cluster]}</p>
                        <p>👥 Cluster Size: {info['size']} customers</p>
                        <p>📅 Average Age: {info['avg_age']} years</p>
                        <p>💰 Average Income: {info['avg_income']:,}</p>
                        <p>🛍️ Average Spending: {info['avg_spending']}</p>
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
        
        # Detailed breakdown
        st.markdown("#### 🔍 Detailed Cluster Breakdown")
        st.dataframe(
            model.df.groupby('Cluster')[['Age', 'Income', 'Spending']].agg(['mean', 'median', 'std']).style.background_gradient()
        )
    
    with tab3:
        st.markdown("### 📈 Visualizations")
        
        # Generate visualizations
        visualizations = model.create_visualizations()
        
        # Display visualizations
        st.pyplot(visualizations['cluster_distribution'])
        st.pyplot(visualizations['income_vs_spending'])
        st.pyplot(visualizations['age_distribution'])
    
    with tab4:
        st.markdown("### 📋 Full Dataset")
        st.dataframe(model.df.style.background_gradient(subset=['Income', 'Spending'])
                     .format({'Cluster': 'Cluster {}'}))

if __name__ == '__main__':
    main()
