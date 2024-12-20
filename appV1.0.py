import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
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
        /* Main Layout Styling */
        .stApp {
            background-color: #f8f9fa;
            font-family: 'Inter', sans-serif;
        }
        
        /* Header Styling */
        .main-header {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .main-header h1 {
            color: #1a237e;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        /* Card Styling */
        .stCard {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Metric Container */
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            margin: 0.5rem 0;
        }
        
        /* Cluster Number Badge */
        .cluster-badge {
            display: inline-block;
            background-color: #1a237e;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-weight: bold;
            margin-right: 0.5rem;
        }
        
        /* Prediction Result */
        .prediction-result {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #1a237e;
            margin: 1rem 0;
        }
        
        /* Cluster Card */
        .cluster-card {
            border-left: 4px solid #1a237e;
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
        }
        
        /* Other styles remain the same */
        .css-1d391kg {
            background-color: #ffffff;
            padding: 2rem 1rem;
        }
        
        .stTextInput > div > div {
            background-color: #ffffff;
            border-radius: 5px;
        }
        
        .stButton > button {
            background-color: #1a237e;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 2rem;
            font-weight: 500;
            width: 100%;
        }
        
        .stButton > button:hover {
            background-color: #283593;
        }
        
        .streamlit-expanderHeader {
            background-color: #ffffff;
            border-radius: 5px;
        }
        
        .streamlit-dataframe {
            border: 1px solid #e9ecef;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

class CustomerSegmentation:
    def __init__(self, csv_path='Customers_Segmentation_with_Clusters.csv'):
        try:
            self.df = pd.read_csv(csv_path)
            self.process_data()
            self.analyze_clusters()
            self.cluster_descriptions = {
                0: "🏦 Conservative Spenders (High Income)",
                1: "⚖️ Balanced Customers",
                2: "💎 Premium Customers",
                3: "⚠️ Risk Group",
                4: "💰 Budget Conscious",
                5: "📈 Moderate Savers"
            }
            self.cluster_details = {
                0: "High income earners with conservative spending habits",
                1: "Customers with balanced earning and spending patterns",
                2: "High-income customers with premium spending habits",
                3: "Lower income group with higher spending patterns",
                4: "Cost-conscious customers with lower income and spending",
                5: "Average income earners with moderate saving habits"
            }
        except FileNotFoundError:
            st.error(f"Error: Could not find {csv_path}")
            
    def process_data(self):
        le = LabelEncoder()
        self.df['Gender_Encoded'] = le.fit_transform(self.df['Gender'])
        self.gender_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        self.features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[self.features])
    
    def analyze_clusters(self):
        self.cluster_info = {}
        for cluster in self.df['Cluster'].unique():
            cluster_data = self.df[self.df['Cluster'] == cluster]
            self.cluster_info[cluster] = {
                'size': len(cluster_data),
                'avg_age': round(cluster_data['Age'].mean(), 1),
                'avg_income': round(cluster_data['Annual Income (k$)'].mean(), 1),
                'avg_spending': round(cluster_data['Spending Score (1-100)'].mean(), 1),
                'gender_distribution': cluster_data['Gender'].value_counts().to_dict()
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
    
    model = CustomerSegmentation()
    
    # Sidebar with improved styling
    with st.sidebar:
        st.markdown("### 📊 Customer Profile Analysis")
        st.markdown("---")
        
        customer_id = st.text_input("📋 Customer ID")
        gender = st.selectbox("👤 Gender", options=["Male", "Female"])
        age = st.number_input("🎂 Age", min_value=0, max_value=100, value=30)
        income = st.number_input("💵 Annual Income (k$)", min_value=0, value=50)
        spending = st.number_input("🛍️ Spending Score (1-100)", min_value=0, max_value=100, value=50)
        
        st.markdown("---")
        if st.button("Analyze Customer"):
            if all([customer_id, gender, age, income, spending]):
                cluster, info = model.predict_segment(customer_id, gender, age, income, spending)
                
                st.markdown(f"""
                    <div class="prediction-result">
                        <h3 style="color: #1a237e;">Predicted Segment</h3>
                        <div class="cluster-card">
                            <span class="cluster-badge">Cluster {cluster}</span>
                            <h4>{model.cluster_descriptions[cluster]}</h4>
                            <p>{model.cluster_details[cluster]}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📊 Detailed Statistics"):
                    st.markdown(f"""
                        <div class="metric-container">
                            <p>👥 Cluster Size: {info['size']} customers</p>
                            <p>📅 Average Age: {info['avg_age']} years</p>
                            <p>💰 Average Income: ${info['avg_income']}k</p>
                            <p>🛍️ Average Spending: {info['avg_spending']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please complete all fields")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Cluster Overview")
        for cluster in sorted(model.cluster_descriptions.keys()):
            with st.expander(f"Cluster {cluster} | {model.cluster_descriptions[cluster]}"):
                info = model.cluster_info[cluster]
                st.markdown(f"""
                    <div class="cluster-card">
                        <span class="cluster-badge">Cluster {cluster}</span>
                        <p>{model.cluster_details[cluster]}</p>
                        <hr>
                        <h4>Key Metrics:</h4>
                        <p>👥 Cluster Size: {info['size']} customers</p>
                        <p>📅 Average Age: {info['avg_age']} years</p>
                        <p>💰 Average Income: ${info['avg_income']}k</p>
                        <p>🛍️ Average Spending Score: {info['avg_spending']}</p>
                        <h4>Gender Distribution:</h4>
                        <p>{'  |  '.join(f'{gender}: {count}' for gender, count in info['gender_distribution'].items())}</p>
                    </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 Data Overview")
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        df_display = model.df.copy()
        st.dataframe(
            df_display.style.background_gradient(subset=['Annual Income (k$)', 'Spending Score (1-100)'])
                          .format({'Cluster': 'Cluster {}'}),
            height=400
        )
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()