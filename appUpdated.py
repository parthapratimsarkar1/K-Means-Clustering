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
        <img src="data:image/jpeg;base64,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" width="200">
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
