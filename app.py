import streamlit as st
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from groq import Groq
import pandas as pd
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from PIL import Image as PILImage

# Enhanced Configuration and Setup
st.set_page_config(
    page_title="Advanced Health Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Expanded Normal Ranges with More Nuanced Categories
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
    'insulin': {
        'low_risk': [(2.6, 24.9)],
        'medium_risk': [(25, 50)],
        'high_risk': [(0, 2.5), (50, 500)]
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
    },
    'cholesterol': {
        'low_risk': [(0, 200)],
        'medium_risk': [(200, 239)],
        'high_risk': [(240, 500)]
    },
    'triglycerides': {
        'low_risk': [(0, 150)],
        'medium_risk': [(150, 199)],
        'high_risk': [(200, 500)]
    }
}

class AdvancedHealthRiskAssessment:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["api_key"])
        except Exception as e:
            st.error(f"API Configuration Error: {e}")
            self.client = None

    def calculate_advanced_risk(self, patient_data):
        """
        Advanced risk calculation with comprehensive multi-factor assessment
        """
        age, glucose, insulin, bmi, systolic_bp, diastolic_bp, cholesterol, triglycerides = patient_data
        
        def calculate_category_risk(value, categories):
            if isinstance(value, tuple):
                systolic, diastolic = value
                for category in categories:
                    if len(category) == 4:
                        min_sys, max_sys, min_dia, max_dia = category
                        if min_sys <= systolic <= max_sys and min_dia <= diastolic <= max_dia:
                            return 0.0
                return 1.0
            else:
                for category in categories:
                    if len(category) == 2:
                        min_val, max_val = category
                        if min_val <= value <= max_val:
                            return 0.0
                return 1.0

        risk_components = {
            'age': calculate_category_risk(age, NORMAL_RANGES['age']['low_risk']),
            'glucose': calculate_category_risk(glucose, NORMAL_RANGES['glucose']['low_risk']),
            'insulin': calculate_category_risk(insulin, NORMAL_RANGES['insulin']['low_risk']),
            'bmi': calculate_category_risk(bmi, NORMAL_RANGES['bmi']['low_risk']),
            'blood_pressure': calculate_category_risk(
                (systolic_bp, diastolic_bp), 
                NORMAL_RANGES['blood_pressure']['low_risk']
            ),
            'cholesterol': calculate_category_risk(cholesterol, NORMAL_RANGES['cholesterol']['low_risk']),
            'triglycerides': calculate_category_risk(triglycerides, NORMAL_RANGES['triglycerides']['low_risk'])
        }

        # Enhanced weighted risk calculation
        weights = {
            'age': 0.15, 
            'glucose': 0.15, 
            'insulin': 0.15,
            'bmi': 0.15, 
            'blood_pressure': 0.1, 
            'cholesterol': 0.15,
            'triglycerides': 0.15
        }
        total_risk = sum(risk * weights[key] for key, risk in risk_components.items())
        
        return min(total_risk, 1.0), risk_components

    def generate_health_recommendations(self, patient_data, risk_score, risk_components):
        """
        Generate comprehensive, personalized health recommendations
        """
        if not self.client:
            return "API configuration error: Unable to generate recommendations."

        patient_name, age, glucose, insulin, bmi, systolic_bp, diastolic_bp, cholesterol, triglycerides = patient_data

        prompt = f"""
        Patient Profile:
        - Name: {patient_name}
        - Age: {age} years
        - Glucose Level: {glucose} mg/dL
        - Insulin Level: {insulin} µIU/mL
        - BMI: {bmi}
        - Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg
        - Cholesterol: {cholesterol} mg/dL
        - Triglycerides: {triglycerides} mg/dL
        - Overall Risk Score: {risk_score:.2f}

        Risk Component Breakdown:
        {', '.join(f"{k.capitalize()}: {'High' if v > 0.5 else 'Low'}" for k, v in risk_components.items())}

        Provide comprehensive, personalized health recommendations addressing:
        1. Specific lifestyle modifications
        2. Dietary suggestions
        3. Exercise recommendations
        4. Potential medical screenings
        5. Mental health considerations
        
        Tailor advice to patient's specific risk factors and metabolic profile.
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
        Create advanced, interactive risk visualizations
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
        fig_components.update_layout(
            xaxis_title='Health Metrics',
            yaxis_title='Risk Level (%)',
            coloraxis_colorbar=dict(title='Risk')
        )

        return fig_gauge, fig_components

def main():
    # Initialize the risk assessment system
    risk_manager = AdvancedHealthRiskAssessment()

    # Professional Styling with Modern Design
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stSidebar {
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Enhanced Streamlit UI
    st.title("🩺 Comprehensive Health Risk Assessment")
    
    # Sidebar with elegant design
    with st.sidebar:
        st.image("logo.png", caption="Advanced Health Risk Analyzer", use_column_width=True)
        st.header("📋 Patient Information")
        
        # Patient Name Input
        patient_name = st.text_input("👤 Patient Name", help="Full name for personalized report")
        
        # Styled input widgets with more metrics
        age = st.slider("👶 Age", 18, 100, 50, help="Your current age")
        glucose = st.slider("🍬 Glucose Level (mg/dL)", 50, 200, 100, help="Fasting glucose level")
        insulin = st.slider("💉 Insulin Level (µIU/mL)", 0.0, 500.0, 25.0, help="Fasting insulin level")
        bmi = st.slider("📏 Body Mass Index (BMI)", 10.0, 40.0, 25.0, help="Body mass index")
        systolic_bp = st.slider("💓 Systolic Blood Pressure", 80, 200, 120, help="Upper blood pressure reading")
        diastolic_bp = st.slider("💓 Diastolic Blood Pressure", 60, 130, 80, help="Lower blood pressure reading")
        cholesterol = st.slider("📊 Total Cholesterol", 100, 500, 200, help="Total cholesterol level")
        triglycerides = st.slider("💧 Triglycerides", 50, 500, 150, help="Triglyceride level")

    # Risk Assessment Button
    if st.button("🔍 Assess Health Risk", use_container_width=True):
        patient_data = [patient_name, age, glucose, insulin, bmi, systolic_bp, diastolic_bp, cholesterol, triglycerides]
        
        # Calculate risk
        risk_score, risk_components = risk_manager.calculate_advanced_risk(patient_data[1:])
        
        # Generate recommendations
        health_advice = risk_manager.generate_health_recommendations(patient_data, risk_score, risk_components)
        
        # Risk Category and Advice
        st.subheader("📊 Risk Assessment Results")
        
        risk_category = (
            "🟢 Low Risk" if risk_score < 0.33 else 
            "🟡 Moderate Risk" if risk_score < 0.66 else 
            "🔴 High Risk"
        )
        
        st.markdown(f"**Patient Name:** {patient_name}")
        st.markdown(f"**Overall Risk Category:** {risk_category}")
        st.markdown(f"**Risk Score:** {risk_score * 100:.2f}%")
        
        # Risk Visualization
        fig_gauge, fig_components = risk_manager.visualize_risk(risk_score, risk_components)
        
        # Results Display
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_components, use_container_width=True)
        
        # Expandable Recommendations
        with st.expander("🩺 Personalized Health Recommendations"):
            st.markdown(health_advice)

    # Additional Resources Section
    st.markdown("## 🌟 Additional Health Resources")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📚 Learn More")
        st.markdown("""
        - Health Screening Guidelines
        - Nutrition Resources
        - Fitness Recommendations
        """)
    
    with col2:
        st.markdown("### 🏥 Professional Consultation")
        st.markdown("""
        - Find Local Healthcare Providers
        - Schedule Health Check-ups
        - Mental Health Support
        """)
    
    with col3:
        st.markdown("### 📱 Health Apps")
        st.markdown("""
        - Fitness Tracking
        - Nutrition Logging
        - Stress Management
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Disclaimer:** This tool provides comprehensive health insights and "
        "should not replace professional medical advice. Always consult "
        "with healthcare professionals for personalized medical guidance."
    )

if __name__ == "__main__":
    main()
