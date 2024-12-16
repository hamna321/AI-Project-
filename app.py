import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from groq import Groq

client = Groq(
    api_key= st.secrets["api_key"],
)
# Define normal ranges for each feature
normal_ranges = {
    'age': (45, 60),  # Assume normal range: ages 45-60
    'glucose': (70, 100),  # Normal glucose range in mg/dL
    'bmi': (18.5, 24.9),  # Normal BMI range
    'systolic_bp': 120,  # Normal systolic BP (mm Hg)
    'diastolic_bp': 80  # Normal diastolic BP (mm Hg)
}

# Function to calculate risk score
def calculate_risk(patient_data, normal_ranges):
    age, glucose, bmi, systolic_bp, diastolic_bp = patient_data
    risk_score = 0

    # Age risk
    age_risk = abs(age - np.mean(normal_ranges['age'])) / (normal_ranges['age'][1] - normal_ranges['age'][0])

    # Glucose risk
    glucose_risk = abs(glucose - normal_ranges['glucose'][1]) / (normal_ranges['glucose'][1] - normal_ranges['glucose'][0])

    # BMI risk
    if bmi < normal_ranges['bmi'][0]:
        bmi_risk = (normal_ranges['bmi'][0] - bmi) / normal_ranges['bmi'][0]
    elif bmi > normal_ranges['bmi'][1]:
        bmi_risk = (bmi - normal_ranges['bmi'][1]) / (40 - normal_ranges['bmi'][1])
    else:
        bmi_risk = 0

    # Blood Pressure risk
    bp_risk = max(abs(systolic_bp - normal_ranges['systolic_bp']) / 40, abs(diastolic_bp - normal_ranges['diastolic_bp']) / 40)

    # Total risk score
    risk_score = age_risk + glucose_risk + bmi_risk + bp_risk
    risk_score = min(1, risk_score)  # Cap the score at 1
    return risk_score

# Streamlit UI for patient data input
def get_patient_data():
    st.sidebar.header("Patient Data Entry 📝")
    age = st.sidebar.slider("👶 Age", 18, 100, 50)
    glucose = st.sidebar.slider("🍩 Glucose Level (mg/dL)", 50, 200, 100)
    bmi = st.sidebar.slider("📏 BMI", 10.0, 40.0, 25.0)
    systolic_bp = st.sidebar.slider("💓 Systolic BP (mm Hg)", 80, 200, 120)
    diastolic_bp = st.sidebar.slider("💓 Diastolic BP (mm Hg)", 60, 130, 80)
    return [age, glucose, bmi, systolic_bp, diastolic_bp]

# Clarifai API for health advice
def get_health_advice(prompt):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )
        
        return chat_completion.choices[0].message.content

# Main application logic
st.title("Health Risk and Recommendations App 💉")
st.markdown("""
    Input patient's health information and receive calculated risk score along with tailored health recommendations.
""")

# Get user input
patient_data = get_patient_data()
risk_score = calculate_risk(patient_data, normal_ranges)

# Construct the prompt for Clarifai GPT-4 model
health_status_prompt = (
    f"Provide health recommendations for a patient with the following attributes:\n"
    f"Age: {patient_data[0]} years, Glucose Level: {patient_data[1]} mg/dL, "
    f"BMI: {patient_data[2]}, Systolic BP: {patient_data[3]} mm Hg, "
    f"Diastolic BP: {patient_data[4]} mm Hg.\n"
    f"Calculated Risk Score: {risk_score:.2f}. Give specific advice on lifestyle, diet, and stress management."
)

# Get advice from Clarifai model
with st.spinner("Generating recommendations..."):
    health_advice = get_health_advice(health_status_prompt)

# Display results
st.subheader(f"📊 Calculated Risk Score: **{risk_score:.2f}**")

# Risk Category
risk_category = "🟢 Low Risk" if risk_score < 0.4 else ("🟡 Medium Risk" if risk_score < 0.7 else "🔴 High Risk")
st.subheader(f"Risk Category: {risk_category}")

# Display Health Recommendations
st.subheader("👨‍⚕️ Tailored Health Recommendations")
st.markdown(health_advice)

# Visualization
fig, ax = plt.subplots()
ax.bar([1, 0], [risk_score, 1 - risk_score], color=["red", "green"])
ax.set_xticks([1, 0])
ax.set_xticklabels([f"Risk: {risk_score * 100:.2f}%", f"Normal: {(1 - risk_score) * 100:.2f}%"])
ax.set_ylabel("Probability")
ax.set_title("Health Risk Breakdown")
st.pyplot(fig)

# Footer
st.markdown("Made with ❤️ for better patient care 👩‍⚕️👨‍⚕️")
