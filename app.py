import streamlit as st
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from groq import Groq
import pandas as pd
import plotly.figure_factory as ff
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor

# Enhanced Configuration and Setup
st.set_page_config(
    page_title="Comprehensive Health Risk Assessment",
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
    # Added new metrics
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
        age, glucose, bmi, systolic_bp, diastolic_bp, cholesterol, triglycerides = patient_data
        
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
            'glucose': 0.2, 
            'bmi': 0.2, 
            'blood_pressure': 0.15, 
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

        patient_name, age, glucose, bmi, systolic_bp, diastolic_bp, cholesterol, triglycerides = patient_data

        prompt = f"""
        Patient Profile:
        - Name: {patient_name}
        - Age: {age} years
        - Glucose Level: {glucose} mg/dL
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
        
        Tailor advice to patient's specific risk factors and profile.
        """

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Recommendation generation error: {e}"

    def create_pdf_report(self, patient_name, patient_data, risk_score, risk_components, health_advice):
        """
        Generate a comprehensive PDF health report
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'Title', 
            parent=styles['Title'], 
            textColor=HexColor('#2C3E50'),
            fontSize=16,
            spaceAfter=12
        )
        
        content = []
        content.append(Paragraph(f"Health Risk Assessment Report for {patient_name}", title_style))
        
        # Patient Details
        details = [
            f"Age: {patient_data[1]} years",
            f"Glucose Level: {patient_data[2]} mg/dL",
            f"BMI: {patient_data[3]}",
            f"Blood Pressure: {patient_data[4]}/{patient_data[5]} mmHg",
            f"Cholesterol: {patient_data[6]} mg/dL",
            f"Triglycerides: {patient_data[7]} mg/dL"
        ]
        
        for detail in details:
            content.append(Paragraph(detail, styles['Normal']))
        
        content.append(Spacer(1, 12))
        content.append(Paragraph(f"Overall Risk Score: {risk_score * 100:.2f}%", styles['Heading2']))
        
        content.append(Paragraph("Risk Component Breakdown:", styles['Heading3']))
        for key, value in risk_components.items():
            risk_level = 'High Risk' if value > 0.5 else 'Low Risk'
            content.append(Paragraph(f"{key.capitalize()}: {risk_level}", styles['Normal']))
        
        content.append(Spacer(1, 12))
        content.append(Paragraph("Personalized Health Recommendations:", styles['Heading3']))
        
        # Split long recommendations into paragraphs
        recommendation_paras = health_advice.split('\n')
        for para in recommendation_paras:
            content.append(Paragraph(para, styles['Normal']))
        
        doc.build(content)
        pdf = buffer.getvalue()
        buffer.close()
        return pdf

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

        # Plotly Radar Chart for Risk Components
        risk_data = [
            {'Component': k.capitalize(), 'Risk': v * 100} 
            for k, v in risk_components.items()
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[100 - (item['Risk']) for item in risk_data],
            theta=[item['Component'] for item in risk_data],
            fill='toself'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title='Risk Component Breakdown'
        )

        return fig_gauge, fig_radar

def main():
    # Initialize the risk assessment system
    risk_manager = AdvancedHealthRiskAssessment()

    # Enhanced Streamlit UI
    st.title("🩺 Comprehensive Health Risk Assessment")
    
    # Sidebar with elegant design
    with st.sidebar:
        st.image("logo.png", caption="Advanced Health Risk Analyzer")
        st.header("📋 Patient Information")
        
        # Patient Name Input
        patient_name = st.text_input("👤 Patient Name", help="Full name for personalized report")
        
        # Styled input widgets with more metrics
        age = st.slider("👶 Age", 18, 100, 50, help="Your current age")
        glucose = st.slider("🍬 Glucose Level (mg/dL)", 50, 200, 100, help="Fasting glucose level")
        bmi = st.slider("📏 Body Mass Index (BMI)", 10.0, 40.0, 25.0, help="Body mass index")
        systolic_bp = st.slider("💓 Systolic Blood Pressure", 80, 200, 120, help="Upper blood pressure reading")
        diastolic_bp = st.slider("💓 Diastolic Blood Pressure", 60, 130, 80, help="Lower blood pressure reading")
        cholesterol = st.slider("📊 Total Cholesterol", 100, 500, 200, help="Total cholesterol level")
        triglycerides = st.slider("💧 Triglycerides", 50, 500, 150, help="Triglyceride level")

    # Risk Assessment Button
    if st.button("🔍 Assess Health Risk", use_container_width=True):
        patient_data = [patient_name, age, glucose, bmi, systolic_bp, diastolic_bp, cholesterol, triglycerides]
        
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
        fig_gauge, fig_radar = risk_manager.visualize_risk(risk_score, risk_components)
        
        # Results Display
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Expandable Recommendations
        with st.expander("🩺 Personalized Health Recommendations"):
            st.markdown(health_advice)
        
        # PDF Report Generation
        pdf_report = risk_manager.create_pdf_report(patient_name, patient_data, risk_score, risk_components, health_advice)
        
        # Download PDF Button
        st.download_button(
            label="📄 Download Full Report",
            data=pdf_report,
            file_name=f"{patient_name}_health_risk_report.pdf",
            mime="application/pdf"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "**Disclaimer:** This tool provides comprehensive health insights and "
        "should not replace professional medical advice. Always consult "
        "with healthcare professionals for personalized medical guidance."
    )

if __name__ == "__main__":
    main()
