import streamlit as st
import pandas as pd
import numpy as np
from sklearn.externals import joblib

st.set_page_config(page_title="Customer Classify App", layout="wide")

@st.cache_data
def load_data():
    # Adjust file reading function based on actual file format
    df = pd.read_excel("Customers_Spending_with_Clusters.xls")  # if it's an Excel file
    return df

def load_model_and_scaler():
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
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

st.title("Customer Classify App")
st.write("Enter customer information to predict their segment")

try:
    players = load_data()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual Income (k$)", min_value=0, max_value=150, value=50)
        score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100, value=50)

    model, scaler = load_model_and_scaler()

    if st.button("Predict Cluster"):
        new_data_point = pd.DataFrame([[age, income, score]], columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"])
        new_data_scaled = scaler.transform(new_data_point)
        cluster = model.predict(new_data_scaled)[0]
        
        with col2:
            st.subheader("Prediction Results")
            st.markdown(f"<div class='flash'><h3>Predicted Customer Segment: Cluster {cluster}</h3></div>", unsafe_allow_html=True)
            
            cluster_data = players[players['Cluster'] == cluster]
            st.write("### Cluster Characteristics:")
            stats = cluster_data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].describe()
            st.dataframe(stats.round(2))
            
            st.write("### Key Insights:")
            st.write(f"- This cluster contains {len(cluster_data)} customers")
            st.write(f"- Average age: {cluster_data['Age'].mean():.1f} years")
            st.write(f"- Average income: ${cluster_data['Annual Income (k$)'].mean():.1f}k")
            st.write(f"- Average spending score: {cluster_data['Spending Score (1-100)'].mean():.1f}")

    with st.expander("View Sample Data"):
        st.dataframe(players.head())
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Please ensure the data file 'Customers_Spending_with_Clusters.xls' is in the correct location.")
