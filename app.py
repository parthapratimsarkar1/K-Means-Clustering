import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Customer Classify App",
    layout="wide"
)

# Load and prepare dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Updated_Mall_Customers_with_Clusters.csv")
    return df

def train_model(data):
    # Select features for clustering
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X = data[features]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KMeans model
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_scaled)
    
    return kmeans, scaler

# CSS for flashing effect
st.markdown("""
    <style>
    .flash {
        animation: flash 1s ease-in-out infinite;
        padding: 10px;
        border-radius: 5px;
    }
    @keyframes flash {
        0% { background-color: #f3e5f5; }
        50% { background-color: #e1bee7; }
        100% { background-color: #f3e5f5; }
    }
    </style>
    """, unsafe_allow_html=True)

# Main app
st.title("Customer Classify App")
st.write("Enter customer information to predict their segment")

# Load data
try:
    players = load_data()
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Input for new data point
    with col1:
        st.subheader("Customer Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual Income (k$)", min_value=0, max_value=150, value=50)
        score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100, value=50)

    # Train model
    model, scaler = train_model(players)

    # Prediction section
    if st.button("Predict Cluster"):
        # Prepare new data point
        new_data_point = pd.DataFrame(
            [[age, income, score]], 
            columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"]
        )
        
        # Scale the new data point
        new_data_scaled = scaler.transform(new_data_point)
        
        # Predict cluster
        cluster = model.predict(new_data_scaled)[0]
        
        with col2:
            st.subheader("Prediction Results")
            # Flashing background effect for prediction result
            st.markdown(f"<div class='flash'><h3>Customer Belong To: Cluster {cluster}</h3></div>", unsafe_allow_html=True)
            
            # Describe the cluster characteristics
            cluster_data = players[players['Cluster'] == cluster]
            
            st.write("### Cluster Characteristics:")
            stats = cluster_data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].describe()
            st.dataframe(stats.round(2))
            
            # Additional insights
            st.write("### Key Insights:")
            st.write(f"- This cluster contains {len(cluster_data)} customers")
            st.write(f"- Average age: {cluster_data['Age'].mean():.1f} years")
            st.write(f"- Average income: ${cluster_data['Annual Income (k$)'].mean():.1f}k")
            st.write(f"- Average spending score: {cluster_data['Spending Score (1-100)'].mean():.1f}")

    # Display sample data
    with st.expander("View Sample Data"):
        st.dataframe(players.head())
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Please ensure the data file 'Updated_Mall_Customers_with_Clusters.csv' is in the correct location.")
