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
            
    def process_data(self):
        """Process and prepare the data"""
        # Convert Gender to numerical
        le = LabelEncoder()
        self.df['Gender_Encoded'] = le.fit_transform(self.df['Gender'])
        self.gender_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        
        # Select features for clustering
        self.features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        
        # Scale the features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[self.features])
    
    def analyze_clusters(self):
        """Analyze existing clusters in the data"""
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
        """Predict customer segment based on input data"""
        input_data = np.array([[age, income, spending]])
        scaled_input = self.scaler.transform(input_data)
        
        # Calculate distances to all cluster centroids
        distances = []
        for cluster in self.df['Cluster'].unique():
            cluster_data = self.df[self.df['Cluster'] == cluster]
            cluster_center = cluster_data[self.features].mean()
            scaled_center = self.scaler.transform([cluster_center])
            distance = np.linalg.norm(scaled_input - scaled_center)
            distances.append((cluster, distance))
        
        # Predict the closest cluster
        predicted_cluster = min(distances, key=lambda x: x[1])[0]
        
        return predicted_cluster, self.cluster_info[predicted_cluster]

def main():
    # Center the title with custom styling
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: #1F1F1F; padding: 1.5rem 0; font-size: 2.5rem;">Customer Segmentation System</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize the model
    model = CustomerSegmentation()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Predict", "Cluster Overview", "Sample Data"])
    
    # Prediction Tab
    with tab1:
        st.header("Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.text_input("Customer ID")
            gender = st.selectbox("Gender", options=["Male", "Female"])
            age = st.number_input("Age", min_value=0, max_value=100)
            
        with col2:
            income = st.number_input("Annual Income (k$)", min_value=0)
            spending = st.number_input("Spending Score (1-100)", min_value=0, max_value=100)
            
        if st.button("Predict Segment"):
            if all([customer_id, gender, age, income, spending]):
                # Make prediction
                cluster, info = model.predict_segment(customer_id, gender, age, income, spending)
                
                st.success(f"Predicted Customer Segment: Cluster {cluster}")
                st.info(f"Cluster Description: {model.cluster_descriptions[cluster]}")
                
                # Display cluster information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Cluster Statistics")
                    st.write(f"Total customers in cluster: {info['size']}")
                    st.write(f"Average age: {info['avg_age']} years")
                    st.write(f"Average income: ${info['avg_income']}k")
                    st.write(f"Average spending score: {info['avg_spending']}")
                
                with col2:
                    st.subheader("Gender Distribution")
                    for gender, count in info['gender_distribution'].items():
                        st.write(f"{gender}: {count} customers")
            else:
                st.warning("Please fill in all fields")

    # Cluster Overview Tab
    with tab2:
        st.header("Cluster Descriptions and Characteristics")
        for cluster, description in model.cluster_descriptions.items():
            with st.expander(f"Cluster {cluster}: {description.split(':')[0]}", expanded=True):
                st.write(description)
                info = model.cluster_info[cluster]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("📊 Statistics")
                    st.write(f"• Size: {info['size']} customers")
                    st.write(f"• Average age: {info['avg_age']} years")
                    st.write(f"• Average income: ${info['avg_income']}k")
                    st.write(f"• Average spending: {info['avg_spending']}")
                
                with col2:
                    st.write("👥 Gender Distribution")
                    for gender, count in info['gender_distribution'].items():
                        st.write(f"• {gender}: {count} customers")

    # Sample Data Tab
    with tab3:
        st.header("Sample Customer Data")
        st.dataframe(model.df)

if __name__ == '__main__':
    main()
