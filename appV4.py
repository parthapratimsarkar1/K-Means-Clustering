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

# Custom CSS for styling
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
            # Validate file existence
            if not os.path.exists(csv_path):
                st.error(f"Error: File {csv_path} not found. Please check the file path.")
                raise FileNotFoundError(f"Could not find {csv_path}")
            
            # Attempt multiple encodings
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
            
            # Normalize column names
            self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
            st.write("Dataset columns after loading:", list(self.df.columns))  # Debug output
            
            # Validate required columns
            required_columns = ['age', 'income_(inr)', 'spending_(1-100)', 'gender', 'cluster']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            self.process_data()
            self.analyze_clusters()
            self.generate_cluster_descriptions()

        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}")
            raise
    
    def process_data(self):
        le = LabelEncoder()
        self.df['gender_encoded'] = le.fit_transform(self.df['gender'])
        self.gender_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        
        self.features = ['age', 'income_(inr)', 'spending_(1-100)']
        self.scaler = StandardScaler()
        
        try:
            self.scaled_features = self.scaler.fit_transform(self.df[self.features])
        except Exception as e:
            st.error(f"Error in data scaling: {e}")
            raise
    
    def generate_cluster_descriptions(self):
        self.cluster_descriptions = {
            -1: "🏦 Noise Points",
             0: "🏦 Conservative Spenders (High Income)",
             1: "⚠️ Risk Customers",
             2: "💎 Premium Customers",
             3: "⚖️ Balanced Group"
        }
        self.cluster_details = {
            -1: "Noise or Outliers",
             0: "Middle Income, Moderate Spending",
             1: "Low Income, Low Spending",
             2: "High Income, High Spending",
             3: "Upper Middle Income, Low Spending"
        }
    
    def analyze_clusters(self):
        unique_clusters = sorted(self.df['cluster'].unique())
        self.cluster_info = {}
        
        for cluster in unique_clusters:
            cluster_data = self.df[self.df['cluster'] == cluster]
            
            if len(cluster_data) > 0:
                self.cluster_info[cluster] = {
                    'size': len(cluster_data),
                    'avg_age': round(cluster_data['age'].mean(), 1),
                    'avg_income': round(cluster_data['income_(inr)'].mean(), 1),
                    'avg_spending': round(cluster_data['spending_(1-100)'].mean(), 1),
                    'gender_distribution': cluster_data['gender'].value_counts().to_dict()
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
        for cluster in self.df['cluster'].unique():
            cluster_data = self.df[self.df['cluster'] == cluster]
            cluster_center = cluster_data[self.features].mean()
            scaled_center = self.scaler.transform([cluster_center])
            distance = np.linalg.norm(scaled_input - scaled_center)
            distances.append((cluster, distance))
        
        predicted_cluster = min(distances, key=lambda x: x[1])[0]
        return predicted_cluster, self.cluster_info[predicted_cluster]

    def create_visualizations(self):
        visualizations = {}

        if PLOTLY_AVAILABLE:
            cluster_sizes = self.df['cluster'].value_counts()
            visualizations['cluster_distribution'] = px.pie(
                values=cluster_sizes.values, 
                names=cluster_sizes.index.map(lambda x: f"Cluster {x}"),
                title="Customer Cluster Distribution",
                hole=0.3
            )

            visualizations['income_vs_spending'] = px.scatter(
                self.df, 
                x='income_(inr)', 
                y='spending_(1-100)', 
                color='cluster',
                title='Income vs Spending by Cluster',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
        else:
            plt.figure(figsize=(10, 6))
            cluster_sizes = self.df['cluster'].value_counts()
            plt.pie(cluster_sizes.values, labels=[f"Cluster {x}" for x in cluster_sizes.index], autopct='%1.1f%%')
            plt.title("Customer Cluster Distribution")
            visualizations['cluster_distribution'] = plt

            plt.figure(figsize=(10, 6))
            for cluster in self.df['cluster'].unique():
                cluster_data = self.df[self.df['cluster'] == cluster]
                plt.scatter(cluster_data['income_(inr)'], cluster_data['spending_(1-100)'], label=f'Cluster {cluster}')
            plt.xlabel('Income (INR)')
            plt.ylabel('Spending Score')
            plt.title('Income vs Spending by Cluster')
            plt.legend()
            visualizations['income_vs_spending'] = plt

        return visualizations

def main():
    st.markdown('<div class="main-header"><h1>Customer Segmentation System</h1></div>', unsafe_allow_html=True)
    
    try:
        model = CustomerSegmentation()
    except Exception as e:
        st.error("Failed to initialize the model. Please check your dataset.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Customer Analysis", "📊 Data Overview", "📈 Visualizations", "📋 Full Dataset"])
    
    with tab1:
        with st.sidebar:
            st.markdown("### 📊 Customer Profile Analysis")
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
    
    with tab2:
        st.markdown("### 📊 Comprehensive Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔢 Basic Statistics")
            st
.write(model.df.describe())

        with col2:
            st.markdown("#### 🔀 Data Information")
            st.write(model.df.info())

    with tab3:
        st.markdown("### 📈 Visualizations")
        visualizations = model.create_visualizations()

        if PLOTLY_AVAILABLE:
            st.plotly_chart(visualizations['cluster_distribution'], use_container_width=True)
            st.plotly_chart(visualizations['income_vs_spending'], use_container_width=True)
        else:
            st.pyplot(visualizations['cluster_distribution'])
            st.pyplot(visualizations['income_vs_spending'])

    with tab4:
        st.markdown("### 📋 Full Dataset")
        st.dataframe(model.df)

if __name__ == "__main__":
    main()
