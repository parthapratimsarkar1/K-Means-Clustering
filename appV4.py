import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Set page configuration
st.set_page_config(
    page_title="DBSCAN Customer Segmentation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

class DBSCANCustomerSegmentation:
    def __init__(self, csv_path='Customers Dataset DBSCAN.csv'):
        # Load the dataset
        self.df = pd.read_csv(csv_path)
        
        # Predefined cluster descriptions
        self.cluster_descriptions = {
            0: {
                'name': '🏦 Conservative Spenders',
                'description': 'High Income, Moderate Spending',
                'color': '#3498db',
                'icon': '💰'
            },
            1: {
                'name': '⚖️ Balanced Customers', 
                'description': 'Moderate Income, Balanced Spending',
                'color': '#2ecc71',
                'icon': '⚖️'
            },
            2: {
                'name': '💎 Premium Customers',
                'description': 'High Income, High Spending',
                'color': '#9b59b6',
                'icon': '💎'
            },
            3: {
                'name': '⚠️ Risk Group',
                'description': 'High Income, Low Spending',
                'color': '#e74c3c',
                'icon': '⚠️'
            },
            -1: {
                'name': '🌟 Unique Outliers',
                'description': 'Non-Standard Customer Profile',
                'color': '#f39c12',
                'icon': '🌟'
            }
        }
        
        # Prepare data for clustering
        self.prepare_data()
    
    def prepare_data(self):
        # Select features for clustering
        features = ['Income (INR)', 'Spending (1-100)']
        
        # Scale the features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(self.df[features])
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.df['Cluster'] = dbscan.fit_predict(scaled_features)
        
        # Optional: Compute cluster centroids
        self.compute_cluster_centroids(scaled_features)
    
    def compute_cluster_centroids(self, scaled_features):
        # Compute cluster centroids
        self.cluster_centroids = {}
        for cluster in self.df['Cluster'].unique():
            mask = self.df['Cluster'] == cluster
            cluster_points = scaled_features[mask]
            centroid = cluster_points.mean(axis=0)
            self.cluster_centroids[cluster] = centroid
    
    def predict_customer_cluster(self, income, spending):
        # Scale the input
        input_data = self.scaler.transform([[income, spending]])
        
        # Compute distances to cluster centroids
        distances = {}
        for cluster, centroid in self.cluster_centroids.items():
            distance = np.linalg.norm(input_data - centroid)
            distances[cluster] = distance
        
        # Find the nearest cluster
        predicted_cluster = min(distances, key=distances.get)
        return predicted_cluster
    
    def analyze_customer_profile(self, customer_id, gender, age, income, spending):
        # Predict cluster
        cluster = self.predict_customer_cluster(income, spending)
        
        return {
            'customer_id': customer_id,
            'cluster': cluster,
            'cluster_details': self.cluster_descriptions[cluster],
            'risk_score': self._calculate_risk_score(cluster, income, spending, age),
            'spending_potential': self._calculate_spending_potential(cluster, income, spending),
            'personalized_insights': self._generate_personalized_insights(cluster, gender, age)
        }
    
    def _calculate_risk_score(self, cluster, income, spending, age):
        # Calculate risk based on cluster and other factors
        base_risk = {
            0: 30,   # Conservative Spenders
            1: 50,   # Balanced Customers
            2: 60,   # Premium Customers
            3: 80,   # Risk Group
            -1: 70   # Outliers
        }
        
        risk_modifier = (age / 100) * 20
        spending_variance = abs(spending - 50)
        
        return min(base_risk.get(cluster, 50) + risk_modifier + (spending_variance / 2), 100)
    
    def _calculate_spending_potential(self, cluster, income, spending):
        # Estimate spending potential
        potential_map = {
            0: 0.4,  # Conservative
            1: 0.6,  # Balanced
            2: 0.9,  # Premium
            3: 0.3,  # Risk Group
            -1: 0.5  # Outliers
        }
        
        base_potential = potential_map.get(cluster, 0.5)
        income_factor = min(income / 100000, 1)
        
        return min(base_potential * income_factor * 100, 100)
    
    def _generate_personalized_insights(self, cluster, gender, age):
        # Generate personalized marketing insights
        age_groups = {
            (18, 30): 'Young Professional',
            (31, 45): 'Mid-Career',
            (46, 60): 'Established Professional',
            (61, 100): 'Senior'
        }
        
        age_group = next(
            group_name for (min_age, max_age), group_name in age_groups.items() 
            if min_age <= age <= max_age
        )
        
        insights = {
            0: f"{gender} {age_group} - Conservative Investor, Prefers Stability",
            1: f"{gender} {age_group} - Balanced Financial Approach",
            2: f"{gender} {age_group} - High-Value Customer, Premium Experiences",
            3: f"{gender} {age_group} - Potential High-Growth Customer",
            -1: f"{gender} {age_group} - Unique Financial Profile"
        }
        
        return insights.get(cluster, "Unique Customer Profile")
    
    def create_visualizations(self):
        # Cluster Distribution
        plt.figure(figsize=(10, 6))
        cluster_counts = self.df['Cluster'].value_counts()
        cluster_counts.plot(kind='bar')
        plt.title('Customer Cluster Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Customers')
        cluster_dist_fig = plt.gcf()
        plt.close()
        
        # Income vs Spending Scatter
        plt.figure(figsize=(10, 6))
        for cluster in self.df['Cluster'].unique():
            cluster_data = self.df[self.df['Cluster'] == cluster]
            plt.scatter(cluster_data['Income (INR)'], cluster_data['Spending (1-100)'], 
                        label=f'Cluster {cluster}', alpha=0.7)
        plt.title('Income vs Spending by Cluster')
        plt.xlabel('Income (INR)')
        plt.ylabel('Spending Score')
        plt.legend()
        income_spending_fig = plt.gcf()
        plt.close()
        
        return {
            'cluster_distribution': cluster_dist_fig,
            'income_vs_spending': income_spending_fig
        }

def main():
    # Main title
    st.markdown('<div class="main-header"><h1>DBSCAN Customer Segmentation</h1></div>', unsafe_allow_html=True)
    
    # Initialize model
    model = DBSCANCustomerSegmentation()
    
    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Customer Analysis", 
        "📊 Cluster Overview", 
        "📈 Visualizations", 
        "📋 Full Dataset"
    ])
    
    with tab1:
        # Customer input sidebar
        with st.sidebar:
            st.markdown("### 📊 Customer Profile Analysis")
            st.markdown("---")
            
            customer_id = st.text_input("📋 Customer ID")
            gender = st.selectbox("👤 Gender", options=["Male", "Female", "Other"])
            age = st.number_input("🎂 Age", min_value=0, max_value=100, value=30)
            income = st.number_input("💵 Income (INR)", min_value=0, value=50000)
            spending = st.number_input("🛍️ Spending (1-100)", min_value=0, max_value=100, value=50)
            
            if st.button("Analyze Customer"):
                analysis_result = model.analyze_customer_profile(customer_id, gender, age, income, spending)
                
                st.markdown(f"""
                    <div class="cluster-card">
                        <span class="cluster-badge" style="background-color: {analysis_result['cluster_details']['color']};">
                            {analysis_result['cluster_details']['icon']} Cluster {analysis_result['cluster']}
                        </span>
                        <h4>{analysis_result['cluster_details']['name']}</h4>
                        <p>{analysis_result['cluster_details']['description']}</p>
                        <hr>
                        <div class="metric-container">
                            <p>📊 Risk Score: {analysis_result['risk_score']:.2f}/100</p>
                            <p>💸 Spending Potential: {analysis_result['spending_potential']:.2f}/100</p>
                            <p>🎯 Insight: {analysis_result['personalized_insights']}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Cluster Overview
        st.markdown("### 📊 Cluster Insights")
        for cluster, details in model.cluster_descriptions.items():
            if cluster != -1:
                with st.expander(f"Cluster {cluster} | {details['name']}"):
                    cluster_data = model.df[model.df['Cluster'] == cluster]
                    st.markdown(f"""
                        <div class="cluster-card">
                            <p>{details['description']}</p>
                            <p>👥 Cluster Size: {len(cluster_data)} customers</p>
                            <p>💰 Average Income: ₹{cluster_data['Income (INR)'].mean():,.2f}</p>
                            <p>🛍️ Average Spending: {cluster_data['Spending (1-100)'].mean():.2f}/100</p>
                        </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 Comprehensive Cluster Breakdown")
        
        # Cluster Statistics
        cluster_stats = model.df.groupby('Cluster')[['Income (INR)', 'Spending (1-100)']].agg(['count', 'mean', 'median'])
        st.dataframe(cluster_stats)
    
    with tab3:
        st.markdown("### 📈 Visualizations")
        
        # Generate and display visualizations
        visualizations = model.create_visualizations()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(visualizations['cluster_distribution'])
        
        with col2:
            st.pyplot(visualizations['income_vs_spending'])
    
    with tab4:
        st.markdown("### 📋 Full Dataset")
        st.dataframe(model.df)

if __name__ == '__main__':
    main()
