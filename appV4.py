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
    def __init__(self, csv_path='Customers Dataset DBSCAN With Cluster.csv'):
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

            # Rename the column to match the correct name
            self.df.rename(columns={'Spending  (1-100)': 'Spending (1-100)'}, inplace=True)

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
           -0: "🏦 Noise Points",
            0: "🏦 Conservative Spenders (High Income)",
            1: "⚠️ Risk Customers",
            2: "💎 Premium Customers",
            3: "⚖️ Balanced Group"
        }
        
        self.cluster_details = {
           -0: "Noise or Outliers",
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

    def create_visualizations(self):
        # Fallback to Matplotlib if Plotly is not available
        visualizations = {}

        if PLOTLY_AVAILABLE:
            # Plotly visualizations
            # 1. Cluster Distribution Pie Chart
            cluster_sizes = self.df['Cluster'].value_counts()
            visualizations['cluster_distribution'] = px.pie(
                values=cluster_sizes.values, 
                names=cluster_sizes.index.map(lambda x: f"Cluster {x}"),
                title="Customer Cluster Distribution",
                hole=0.3
            )

            # 2. Income vs Spending Scatter Plot
            visualizations['income_vs_spending'] = px.scatter(
                self.df, 
                x='Income (INR)', 
                y='Spending (1-100)', 
                color='Cluster',
                title='Income vs Spending by Cluster',
                labels={'Income (INR)': 'Income (INR)', 'Spending (1-100)': 'Spending Score'},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
        else:
            # Matplotlib fallback visualizations
            # 1. Cluster Distribution Pie Chart
            plt.figure(figsize=(10, 6))
            cluster_sizes = self.df['Cluster'].value_counts()
            plt.pie(cluster_sizes.values, labels=[f"Cluster {x}" for x in cluster_sizes.index], autopct='%1.1f%%')
            plt.title("Customer Cluster Distribution")
            visualizations['cluster_distribution'] = plt

            # 2. Income vs Spending Scatter Plot
            plt.figure(figsize=(10, 6))
            for cluster in self.df['Cluster'].unique():
                cluster_data = self.df[self.df['Cluster'] == cluster]
                plt.scatter(cluster_data['Income (INR)'], cluster_data['Spending (1-100)'], 
                            label=f'Cluster {cluster}')
            plt.xlabel('Income (INR)')
            plt.ylabel('Spending Score')
            plt.title('Income vs Spending by Cluster')
            plt.legend()
            visualizations['income_vs_spending'] = plt

        # 3. Age Distribution (works with both Plotly and Matplotlib)
        plt.figure(figsize=(10, 6))
        for cluster in self.df['Cluster'].unique():
            cluster_data = self.df[self.df['Cluster'] == cluster]['Age']
            plt.hist(cluster_data, alpha=0.5, label=f'Cluster {cluster}')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution Across Clusters')
        plt.legend()
        visualizations['age_distribution'] = plt

        return visualizations

def main():
    # Header with professional styling
    st.markdown('<div class="main-header"><h1>Customer Segmentation System</h1></div>', unsafe_allow_html=True)
    
    try:
        model = CustomerSegmentation()
    except Exception as e:
        st.error("Failed to initialize the model. Please check your dataset.")
        return
    
    # Main tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Customer Analysis", "📊 Data Overview", "📈 Visualizations", "📋 Full Dataset"])
    
    with tab1:
        # Sidebar with customer inputs
        with st.sidebar:
            st.markdown("### 📊 Customer Profile Analysis")
            st.markdown("---")
            
            customer_id = st.text_input("📋 Customer ID")
            gender = st.selectbox("👤 Gender", options=["Male", "Female"])
            age = st.number_input("🎂 Age", min_value=0, max_value=100, value=30)
            income = st.number_input("💸 Income (INR)", min_value=0, value=50000)
            spending = st.number_input("💳 Spending Score (1-100)", min_value=0, max_value=100, value=50)
            
            # Predict the cluster
            if customer_id:
                gender_encoded = 0 if gender == "Male" else 1
                predicted_cluster, cluster_info = model.predict_segment(customer_id, gender, age, income, spending)
                st.markdown(f"**Predicted Cluster:** {predicted_cluster} ({model.cluster_descriptions.get(predicted_cluster, 'No description available')})")
                
                # Display cluster details
                st.markdown(f"**Cluster Size:** {cluster_info['size']}")
                st.markdown(f"**Average Age:** {cluster_info['avg_age']}")
                st.markdown(f"**Average Income (INR):** {cluster_info['avg_income']}")
                st.markdown(f"**Average Spending Score:** {cluster_info['avg_spending']}")
                
                st.markdown("### Gender Distribution in Cluster:")
                for gender, count in cluster_info['gender_distribution'].items():
                    st.markdown(f"- {gender}: {count} members")
                
    with tab2:
        st.write("Displaying basic data overview")
        st.write(model.df.head())

    with tab3:
        st.write("Visualizing cluster data")
        visualizations = model.create_visualizations()
        st.pyplot(visualizations['age_distribution'])
        if PLOTLY_AVAILABLE:
            st.plotly_chart(visualizations['cluster_distribution'])
            st.plotly_chart(visualizations['income_vs_spending'])
        else:
            st.pyplot(visualizations['cluster_distribution'])
            st.pyplot(visualizations['income_vs_spending'])

    with tab4:
        st.write("Viewing full dataset")
        st.write(model.df)
