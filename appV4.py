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
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not installed. Using Matplotlib for visualizations.")

warnings.filterwarnings('ignore')

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

class CustomerSegmentation:
    def __init__(self, csv_path='Customer-Dataset-With-ClusteredDB.csv'):
        try:
            if not os.path.exists(csv_path):
                st.error(f"Error: File {csv_path} not found. Please check the file path.")
                raise FileNotFoundError(f"Could not find {csv_path}")

            # Attempt to read CSV with multiple encodings
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

            # Filter required columns
            self.df = self.df[['Age', 'Income (INR)', 'Spending (1-100)', 'Gender', 'Cluster']]

            # Check for missing columns
            required_columns = ['Age', 'Income (INR)', 'Spending (1-100)', 'Gender', 'Cluster']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")

            self.process_data()
            self.analyze_clusters()

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

    def analyze_clusters(self):
        self.cluster_centers = self.df.groupby('Cluster')[self.features].mean()
        unique_clusters = sorted(self.df['Cluster'].unique())
        self.cluster_info = {}

        for cluster in unique_clusters:
            cluster_data = self.df[self.df['Cluster'] == cluster]
            self.cluster_info[cluster] = {
                'size': len(cluster_data),
                'avg_age': round(cluster_data['Age'].mean(), 1),
                'avg_income': round(cluster_data['Income (INR)'].mean(), 1),
                'avg_spending': round(cluster_data['Spending (1-100)'].mean(), 1),
                'gender_distribution': cluster_data['Gender'].value_counts().to_dict()
            }

    def predict_segment(self, customer_id, gender, age, income, spending):
        input_data = np.array([[age, income, spending]])
        scaled_input = self.scaler.transform(input_data)

        distances = self.cluster_centers.apply(
            lambda center: np.linalg.norm(scaled_input - self.scaler.transform([center.values])),
            axis=1
        )
        predicted_cluster = distances.idxmin()
        return predicted_cluster, self.cluster_info[predicted_cluster]

    def create_visualizations(self):
        visualizations = {}
        if PLOTLY_AVAILABLE:
            cluster_sizes = self.df['Cluster'].value_counts()
            visualizations['cluster_distribution'] = px.pie(
                values=cluster_sizes.values,
                names=cluster_sizes.index.map(lambda x: f"Cluster {x}"),
                title="Customer Cluster Distribution"
            )
        else:
            plt.figure(figsize=(10, 6))
            cluster_sizes = self.df['Cluster'].value_counts()
            plt.pie(cluster_sizes.values, labels=[f"Cluster {x}" for x in cluster_sizes.index], autopct='%1.1f%%')
            plt.title("Customer Cluster Distribution")
            visualizations['cluster_distribution'] = plt
        return visualizations

def main():
    st.markdown('<div class="main-header"><h1>Customer Segmentation System</h1></div>', unsafe_allow_html=True)

    try:
        model = CustomerSegmentation()
    except Exception as e:
        st.error("Failed to initialize the model. Please check your dataset.")
        return

    tab1, tab2, tab3 = st.tabs(["🔍 Customer Analysis", "📈 Visualizations", "📋 Full Dataset"])

    with tab1:
        with st.sidebar:
            st.markdown("### Customer Profile Analysis")
            customer_id = st.text_input("Customer ID")
            gender = st.selectbox("Gender", options=["Male", "Female"])
            age = st.number_input("Age", min_value=0, max_value=100, value=30)
            income = st.number_input("Income (INR)", min_value=0, value=50000)
            spending = st.number_input("Spending (1-100)", min_value=0, max_value=100, value=50)

            if st.button("Analyze Customer"):
                if all([customer_id, gender, age, income, spending]):
                    try:
                        cluster, info = model.predict_segment(customer_id, gender, age, income, spending)
                        st.markdown(f"""
                            <div class="cluster-card">
                                <span class="cluster-badge">Cluster {cluster}</span>
                                <h4>{cluster}</h4>
                                <p>Size: {info['size']}</p>
                                <p>Avg Age: {info['avg_age']}</p>
                                <p>Avg Income: ₹{info['avg_income']}</p>
                                <p>Avg Spending: {info['avg_spending']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error in customer analysis: {e}")

    with tab2:
        visualizations = model.create_visualizations()
        st.plotly_chart(visualizations['cluster_distribution'])

    with tab3:
        st.dataframe(model.df)

if __name__ == '__main__':
    main()
