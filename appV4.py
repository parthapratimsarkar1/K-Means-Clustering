import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CustomerSegmentationAnalysis:
    def __init__(self, csv_path='Customers Dataset DBSCAN.csv'):
        # Load the dataset
        self.df = pd.read_csv(csv_path)
        
        # Predefined cluster descriptions
        self.cluster_descriptions = {
            -1: "🚨 Noise Points (Outliers)",
            0: "💼 Middle Income, Moderate Spending",
            1: "📉 Low Income, Low Spending",
            2: "💎 High Income, High Spending",
            3: "💰 Upper Middle Income, Low Spending"
        }
        
        # Cluster characteristics
        self.cluster_characteristics = {
            -1: {"Mean_Income": "Varies", "Mean_Spending": "Varies"},
            0: {"Mean_Income": 49733, "Mean_Spending": 52.79},
            1: {"Mean_Income": 24583, "Mean_Spending": 9.58},
            2: {"Mean_Income": 80375, "Mean_Spending": 82.94},
            3: {"Mean_Income": 83423, "Mean_Spending": 13.77}
        }

    def analyze_customer_profile(self, gender, age, income, spending):
        """
        Analyze a customer's profile based on input parameters
        """
        # Find the closest cluster based on income and spending
        distances = {}
        for cluster, chars in self.cluster_characteristics.items():
            if cluster == -1:  # Skip noise cluster
                continue
            
            # Calculate normalized distance
            income_diff = abs(income - chars['Mean_Income']) / 100000  # Normalize income
            spending_diff = abs(spending - chars['Mean_Spending']) / 100  # Normalize spending
            
            # Weighted distance calculation
            distance = np.sqrt((income_diff * 0.6)**2 + (spending_diff * 0.4)**2)
            distances[cluster] = distance
        
        # Find the closest cluster
        closest_cluster = min(distances, key=distances.get)
        
        # Additional profile insights
        profile_insights = {
            "Cluster": closest_cluster,
            "Cluster Description": self.cluster_descriptions[closest_cluster],
            "Income Comparison": self._compare_income(income),
            "Spending Comparison": self._compare_spending(spending),
            "Age Group": self._categorize_age(age),
            "Potential Marketing Segment": self._get_marketing_segment(closest_cluster, gender, age)
        }
        
        return profile_insights

    def _compare_income(self, income):
        """Compare income to cluster averages"""
        if income < 30000:
            return "Low Income Range"
        elif 30000 <= income < 50000:
            return "Lower Middle Income Range"
        elif 50000 <= income < 80000:
            return "Middle Income Range"
        elif 80000 <= income < 100000:
            return "Upper Middle Income Range"
        else:
            return "High Income Range"

    def _compare_spending(self, spending):
        """Compare spending to cluster averages"""
        if spending < 20:
            return "Very Low Spending"
        elif 20 <= spending < 40:
            return "Low Spending"
        elif 40 <= spending < 60:
            return "Moderate Spending"
        elif 60 <= spending < 80:
            return "High Spending"
        else:
            return "Very High Spending"

    def _categorize_age(self, age):
        """Categorize age groups"""
        if age < 25:
            return "Young Adult"
        elif 25 <= age < 40:
            return "Working Professional"
        elif 40 <= age < 55:
            return "Established Adult"
        else:
            return "Senior"

    def _get_marketing_segment(self, cluster, gender, age):
        """Determine marketing segment based on cluster, gender, and age"""
        age_group = self._categorize_age(age)
        
        segment_mapping = {
            0: {
                "Young Adult": "Value-Conscious Millennial",
                "Working Professional": "Balanced Spender",
                "Established Adult": "Steady Consumer",
                "Senior": "Conservative Moderate Spender"
            },
            1: {
                "Young Adult": "Budget-Conscious Student",
                "Working Professional": "Frugal Professional",
                "Established Adult": "Cost-Sensitive Family",
                "Senior": "Fixed Income Saver"
            },
            2: {
                "Young Adult": "Luxury-Seeking Millennial",
                "Working Professional": "Premium Lifestyle Professional",
                "Established Adult": "High-End Consumer",
                "Senior": "Affluent Retiree"
            },
            3: {
                "Young Adult": "Strategic Saver",
                "Working Professional": "Investment-Focused Professional",
                "Established Adult": "High Income, Low Consumption",
                "Senior": "Wealth Accumulator"
            }
        }
        
        return f"{gender} {segment_mapping.get(cluster, {}).get(age_group, 'Undefined Segment')}"

def main():
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .result-card {
            background-color: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<div class="main-header"><h1>📊 Customer Profile Analysis</h1></div>', unsafe_allow_html=True)

    # Initialize the analysis model
    model = CustomerSegmentationAnalysis()

    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("👤 Gender", ["Male", "Female", "Other"])
        age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30)
    
    with col2:
        income = st.number_input("💵 Income (INR)", min_value=1000, max_value=1000000, value=50000, step=1000)
        spending = st.number_input("🛍️ Spending Score (1-100)", min_value=1, max_value=100, value=50)

    # Analyze button
    if st.button("🔍 Analyze Customer Profile"):
        # Perform analysis
        profile = model.analyze_customer_profile(gender, age, income, spending)
        
        # Display results
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        st.markdown(f"""
        ### 📊 Customer Segmentation Analysis
        
        **Cluster:** {profile['Cluster']} - {profile['Cluster Description']}
        
        **Income Profile:** {profile['Income Comparison']}
        
        **Spending Behavior:** {profile['Spending Comparison']}
        
        **Age Group:** {profile['Age Group']}
        
        **Marketing Segment:** {profile['Potential Marketing Segment']}
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
