import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path='Mall_Customers.csv'):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Select features for clustering
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=4)
    labels = dbscan.fit_predict(X_scaled)
    
    return data, scaler, dbscan

def predict_cluster(input_data, scaler, dbscan, original_data):
    # Prepare input data for clustering
    input_features = np.array([[input_data['Annual Income (k$)'], 
                                input_data['Spending Score (1-100)']]])
    
    # Scale the input features
    input_scaled = scaler.transform(input_features)
    
    # Combine input with original data for clustering
    X_original = original_data[['Annual Income (k$)', 'Spending Score (1-100)']].values
    X_original_scaled = scaler.transform(X_original)
    
    # Combine scaled data
    X_combined = np.vstack([X_original_scaled, input_scaled])
    
    # Refit DBSCAN with combined data
    combined_labels = dbscan.fit_predict(X_combined)
    
    # The last label is the cluster for the new point
    cluster = combined_labels[-1]
    
    return cluster

def main():
    st.title('Mall Customer Clustering App')
    
    # Load data and prepare clustering model
    data, scaler, dbscan = load_and_preprocess_data()
    
    # Sidebar for input
    st.sidebar.header('Customer Details')
    
    # Input fields
    customer_id = st.sidebar.number_input('Customer ID', min_value=1, value=1)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    age = st.sidebar.number_input('Age', min_value=1, max_value=100, value=30)
    annual_income = st.sidebar.number_input('Annual Income (k$)', min_value=1, max_value=200, value=50)
    spending_score = st.sidebar.number_input('Spending Score (1-100)', min_value=1, max_value=100, value=50)
    
    # Predict Cluster
    if st.sidebar.button('Predict Cluster'):
        input_data = {
            'Annual Income (k$)': annual_income,
            'Spending Score (1-100)': spending_score
        }
        
        # Predict cluster
        cluster = predict_cluster(input_data, scaler, dbscan, data)
        
        # Display results
        st.header('Clustering Results')
        
        # Cluster interpretation
        if cluster == -1:
            st.warning('Customer Profile is Unique')
            st.write('This customer does not fit into any existing cluster patterns.')
            
            # Additional insights for unique customers
            st.subheader('Unique Customer Insights')
            
            # Compare input to overall dataset statistics
            income_mean = data['Annual Income (k$)'].mean()
            income_std = data['Annual Income (k$)'].std()
            spending_mean = data['Spending Score (1-100)'].mean()
            spending_std = data['Spending Score (1-100)'].std()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric('Annual Income', 
                          f"${annual_income}k", 
                          delta=f"{(annual_income - income_mean)/income_std:.2f} std from mean")
            with col2:
                st.metric('Spending Score', 
                          f"{spending_score}", 
                          delta=f"{(spending_score - spending_mean)/spending_std:.2f} std from mean")
            
            # Visualize position relative to other clusters
            st.subheader('Customer Position Relative to Clusters')
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                data['Annual Income (k$)'], 
                data['Spending Score (1-100)'], 
                c=data['Cluster'], 
                cmap='viridis'
            )
            ax.scatter(
                annual_income, 
                spending_score, 
                color='red', 
                marker='x', 
                s=200, 
                label='New Customer'
            )
            ax.set_xlabel('Annual Income (k$)')
            ax.set_ylabel('Spending Score (1-100)')
            ax.set_title('Customer Positioning')
            plt.colorbar(scatter, label='Cluster')
            ax.legend()
            st.pyplot(fig)
        else:
            st.success(f'Cluster {cluster}: Customer belongs to Cluster {cluster}')
            
            # Cluster characteristics
            cluster_data = data[data['Cluster'] == cluster]
            st.subheader('Cluster Characteristics')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Avg Annual Income', 
                          f"${cluster_data['Annual Income (k$)'].mean():.2f}k")
            with col2:
                st.metric('Avg Spending Score', 
                          f"{cluster_data['Spending Score (1-100)'].mean():.2f}")
            with col3:
                st.metric('Avg Age', 
                          f"{cluster_data['Age'].mean():.2f}")
    
    # Optional: Show full cluster summary
    if st.checkbox('Show Full Cluster Summary'):
        cluster_summary = data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']].mean()
        st.dataframe(cluster_summary)

if __name__ == '__main__':
    main()
