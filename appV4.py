import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

class CustomerSegmentationAnalysis:
    def __init__(self, csv_path='Customers Dataset DBSCAN.csv'):
        # Load the dataset
        self.df = pd.read_csv(csv_path)
        
        # Predefined cluster characteristics
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

    def analyze_customer_profile(self, customer_id, gender, age, income, spending):
        # Simulate cluster assignment and analysis
        cluster = self._predict_cluster(income, spending)
        
        return {
            'customer_id': customer_id,
            'cluster': cluster,
            'cluster_details': self.cluster_descriptions[cluster],
            'risk_score': self._calculate_risk_score(cluster, income, spending, age),
            'spending_potential': self._calculate_spending_potential(cluster, income, spending),
            'personalized_insights': self._generate_personalized_insights(cluster, gender, age)
        }

    def _predict_cluster(self, income, spending):
        # Simple clustering logic based on income and spending
        if income > 80000 and spending < 30:
            return 3  # Risk Group
        elif income > 80000 and spending > 70:
            return 2  # Premium Customers
        elif income > 40000 and 30 <= spending <= 70:
            return 1  # Balanced Customers
        elif income <= 40000:
            return 0  # Conservative Spenders
        else:
            return -1  # Outliers

    def _calculate_risk_score(self, cluster, income, spending, age):
        # Generate a risk score based on multiple factors
        base_risk = {
            0: 30,   # Conservative Spenders
            1: 50,   # Balanced Customers
            2: 60,   # Premium Customers
            3: 80,   # Risk Group
            -1: 70   # Outliers
        }
        
        risk_modifier = (age / 100) * 20  # Age impacts risk
        spending_variance = abs(spending - 50)  # Distance from average spending
        
        return min(base_risk.get(cluster, 50) + risk_modifier + (spending_variance / 2), 100)

    def _calculate_spending_potential(self, cluster, income, spending):
        # Estimate future spending potential
        potential_map = {
            0: 0.4,  # Conservative
            1: 0.6,  # Balanced
            2: 0.9,  # Premium
            3: 0.3,  # Risk Group
            -1: 0.5  # Outliers
        }
        
        base_potential = potential_map.get(cluster, 0.5)
        income_factor = min(income / 100000, 1)  # Normalize income
        
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

def main():
    # Page configuration
    st.set_page_config(
        page_title="Customer Segmentation System", 
        page_icon="📊", 
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main-container {
            background-color: #f4f6f9;
            padding: 20px;
            border-radius: 15px;
        }
        .analysis-card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .cluster-badge {
            display: inline-block;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #e9ecef;
            border-radius: 10px;
            margin: 5px;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main title
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>📊 Customer Segmentation System</h1>", unsafe_allow_html=True)

    # Initialize analysis model
    model = CustomerSegmentationAnalysis()

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Customer Analysis", 
        "📊 Data Overview", 
        "📈 Visualizations", 
        "📋 Full Dataset", 
        "📊 Cluster Overview"
    ])

    with tab1:
        st.markdown("### 📋 Customer Profile Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.text_input("📋 Customer ID", value=f"CUS-{random.randint(1000, 9999)}")
            gender = st.selectbox("👤 Gender", ["Male", "Female", "Other"])
        
        with col2:
            age = st.slider("🎂 Age", 18, 100, 35)
            income = st.number_input("💵 Income (INR)", min_value=1000, max_value=1000000, value=50000, step=1000)
        
        spending = st.slider("🛍️ Spending Score (1-100)", 1, 100, 50)
        
        if st.button("🔍 Analyze Customer"):
            # Perform analysis
            analysis_result = model.analyze_customer_profile(customer_id, gender, age, income, spending)
            
            # Display results
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            
            # Cluster Badge
            st.markdown(f"""
            <div class="cluster-badge" style="background-color: {analysis_result['cluster_details']['color']}; color: white;">
                {analysis_result['cluster_details']['icon']} {analysis_result['cluster_details']['name']}
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Customer ID", analysis_result['customer_id'])
                st.metric("Cluster", analysis_result['cluster_details']['name'])
                st.metric("Cluster Description", analysis_result['cluster_details']['description'])
            
            with col2:
                st.metric("Risk Score", f"{analysis_result['risk_score']:.2f}/100")
                st.metric("Spending Potential", f"{analysis_result['spending_potential']:.2f}/100")
                st.metric("Personalized Insight", analysis_result['personalized_insights'])
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Placeholder tabs for future implementation
    with tab2:
        st.markdown("### 📊 Data Overview")
        st.dataframe(model.df.describe())

    with tab3:
        st.markdown("### 📈 Visualizations")
        # Add visualization logic here

    with tab4:
        st.markdown("### 📋 Full Dataset")
        st.dataframe(model.df)

    with tab5:
        st.markdown("### 📊 Cluster Overview")
        for cluster, details in model.cluster_descriptions.items():
            if cluster != -1:
                st.markdown(f"""
                ### {details['icon']} {details['name']}
                **Description:** {details['description']}
                """)

if __name__ == '__main__':
    main()
