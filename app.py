import streamlit as st
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from groq import Groq

# Enhanced Configuration and Setup
st.set_page_config(
    page_title="Advanced Health Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Improved Normal Ranges with More Nuanced Categories
NORMAL_RANGES = {
    'age': {
        'low_risk': [(45, 55)],
        'medium_risk': [(35, 65)],
        'high_risk': [(0, 35), (65, 100)]
    },
    'glucose': {
        'low_risk': [(70, 100)],
        'medium_risk': [(100, 125)],
        'high_risk': [(125, 200)]
    },
    'bmi': {
        'low_risk': [(18.5, 24.9)],
        'medium_risk': [(25, 29.9), (15, 18.5)],
        'high_risk': [(30, 40), (0, 15)]
    },
    'blood_pressure': {
        'low_risk': [(90, 120, 60, 80)],
        'medium_risk': [(121, 139, 81, 89)],
        'high_risk': [(140, 200, 90, 120)]
    }
}

class HealthRiskAssessment:
    def __init__(self):
        # Initialize Groq Client with error handling
        try:
            self.client = Groq(api_key=st.secrets["api_key"])
        except Exception as e:
            st.error(f"API Configuration Error: {e}")
            self.client = None

    def calculate_advanced_risk(self, patient_data):
        """
        Advanced risk calculation with weighted scoring and multi-tier risk assessment
        """
        age, glucose, bmi, systolic_bp, diastolic_bp = patient_data
        
        def calculate_category_risk(value, categories):
            # Handle different types of risk categories
            if isinstance(value, tuple):  # Blood pressure case
                systolic, diastolic = value
                for category in categories:
                    if len(category) == 4:
                        min_sys, max_sys, min_dia, max_dia = category
                        if min_sys <= systolic <= max_sys and min_dia <= diastolic <= max_dia:
                            return 0.0
                return 1.0
            else:  # Single value categories like age, glucose, BMI
                for category in categories:
                    if len(category) == 2:
                        min_val, max_val = category
                        if min_val <= value <= max_val:
                            return 0.0
                return 1.0

        risk_components = {
            'age': calculate_category_risk(age, NORMAL_RANGES['age']['low_risk']),
            'glucose': calculate_category_risk(glucose, NORMAL_RANGES['glucose']['low_risk']),
            'bmi': calculate_category_risk(bmi, NORMAL_RANGES['bmi']['low_risk']),
            'blood_pressure': calculate_category_risk(
                (systolic_bp, diastolic_bp), 
                NORMAL_RANGES['blood_pressure']['low_risk']
            )
        }

        # Weighted risk calculation
        weights = {'age': 0.2, 'glucose': 0.3, 'bmi': 0.25, 'blood_pressure': 0.25}
        total_risk = sum(risk * weights[key] for key, risk in risk_components.items())
        
        return min(total_risk, 1.0), risk_components

    def generate_health_recommendations(self, patient_data, risk_score, risk_components):
        """
        Generate contextual health recommendations
        """
        if not self.client:
            return "API configuration error: Unable to generate recommendations."

        prompt = f"""
        Patient Profile:
        - Age: {patient_data[0]} years
        - Glucose Level: {patient_data[1]} mg/dL
        - BMI: {patient_data[2]}
        - Blood Pressure: {patient_data[3]}/{patient_data[4]} mmHg
        - Overall Risk Score: {risk_score:.2f}

        Risk Component Breakdown:
        {', '.join(f"{k.capitalize()}: {'High' if v > 0.5 else 'Low'}" for k, v in risk_components.items())}

        Provide highly personalized, actionable health recommendations addressing specific risk areas.
        Use a professional medical tone, include specific lifestyle modifications, potential screenings, 
        and preventive strategies.
        """

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Recommendation generation error: {e}"

    def visualize_risk(self, risk_score, risk_components):
        """
        Create interactive, informative risk visualization
        """
        # Plotly Gauge Chart for Overall Risk
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Health Risk Score", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "red"}
                ],
            }
        ))

        # Plotly Bar Chart for Risk Components
        risk_data = [
            {'Component': k.capitalize(), 'Risk': v * 100} 
            for k, v in risk_components.items()
        ]
        
        fig_components = px.bar(
            risk_data, 
            x='Component', 
            y='Risk', 
            title='Risk Component Breakdown',
            color='Risk',
            color_continuous_scale='RdYlGn_r'
        )

        return fig_gauge, fig_components

def main():
    # Initialize the risk assessment system
    risk_manager = HealthRiskAssessment()

    # Enhanced Streamlit UI
    st.title("🩺 Comprehensive Health Risk Assessment")
    
    # Sidebar with elegant design
    with st.sidebar:
        st.image("https://via.placeholder.com/150", caption="Health Risk Analyzer")
        st.header("📋 Patient Information")
        
        # Styled input widgets
        age = st.slider("👶 Age", 18, 100, 50, help="Your current age")
        glucose = st.slider("🍬 Glucose Level (mg/dL)", 50, 200, 100, help="Fasting glucose level")
        bmi = st.slider("📏 Body Mass Index (BMI)", 10.0, 40.0, 25.0, help="Body mass index")
        systolic_bp = st.slider("💓 Systolic Blood Pressure", 80, 200, 120, help="Upper blood pressure reading")
        diastolic_bp = st.slider("💓 Diastolic Blood Pressure", 60, 130, 80, help="Lower blood pressure reading")

    # Risk Assessment Button
    if st.button("🔍 Assess Health Risk", use_container_width=True):
        patient_data = [age, glucose, bmi, systolic_bp, diastolic_bp]
        
        # Calculate risk
        risk_score, risk_components = risk_manager.calculate_advanced_risk(patient_data)
        
        # Generate recommendations
        health_advice = risk_manager.generate_health_recommendations(patient_data, risk_score, risk_components)
        
        # Risk Visualization
        fig_gauge, fig_components = risk_manager.visualize_risk(risk_score, risk_components)
        
        # Results Display
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_components, use_container_width=True)
        
        # Risk Category and Advice
        st.subheader("📊 Risk Assessment Results")
        
        risk_category = (
            "🟢 Low Risk" if risk_score < 0.33 else 
            "🟡 Moderate Risk" if risk_score < 0.66 else 
            "🔴 High Risk"
        )
        
        st.markdown(f"**Overall Risk Category:** {risk_category}")
        st.markdown(f"**Risk Score:** {risk_score * 100:.2f}%")
        
        # Expandable Recommendations
        with st.expander("🩺 Personalized Health Recommendations"):
            st.markdown(health_advice)

    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This tool provides general health insights and should not replace professional medical advice.")

if __name__ == "__main__":
    main()
