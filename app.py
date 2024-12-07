import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import streamlit as st


# Load Pima Indians Diabetes Dataset
diabetes_data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', header=None)

# Assign column names
columns = ['Pregnancies 🤰', 'Glucose 🩸', 'BloodPressure 💓', 'SkinThickness 🖐️', 'Insulin 💉', 
           'BMI ⚖️', 'DiabetesPedigreeFunction 🧬', 'Age 👶', 'Outcome 🏥']
diabetes_data.columns = columns

# Features and labels
X = diabetes_data.drop('Outcome 🏥', axis=1)
y = diabetes_data['Outcome 🏥']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)


# Function to suggest health advice
def give_suggestion(prediction, user_data):
    """
    Generate suggestions based on prediction and user's health data.
    Args:
        prediction: Risk prediction (1 means at risk, 0 means low risk).
        user_data: Dictionary - Contains the user's entered health information.
    Returns:
        Health advice string.
    """
    # Risky health condition warnings
    if prediction == 1:
        health_advice = ""
        if user_data['BMI ⚖️'] > 25:
            health_advice += "⚖️ Your BMI is above the normal range. Focus on a healthy diet and exercise.\n"
        if user_data['Glucose 🩸'] > 140:
            health_advice += "🩸 Your blood glucose is high. Monitor sugar intake.\n"
        if user_data['BloodPressure 💓'] > 130:
            health_advice += "💓 Your blood pressure is elevated. Reduce stress and monitor diet.\n"
        return f"""
        🚨 **You are at risk of developing diabetes. Here are some tips to stay healthy:**  
        {health_advice}
        🍏 Eat healthy, stay active, and check your health regularly.  
        🏃‍♂️ Incorporate at least 30 minutes of daily exercise.  
        🩺 Visit your healthcare provider for regular check-ups.
        """
    else:
        health_advice = ""
        if user_data['BMI ⚖️'] > 25:
            health_advice += "⚖️ Your BMI is slightly high. Stay consistent with your fitness journey.\n"
        return f"""
        🟢 **You are at a lower risk of developing diabetes. Keep maintaining a healthy lifestyle!**  
        {health_advice}
        🥗 Eat a balanced diet and incorporate fruits and vegetables.  
        🚶‍♀️ Engage in daily physical activity.  
        🏆 Keep up the good work with positive lifestyle choices!
        """


# Function to visualize patient's health vs normal ranges
def plot_patient_health(user_data, diabetes_data):
    """
    Function to visualize comparison between patient health data and population ranges.
    Args:
        user_data: The user's health data entered by them.
        diabetes_data: The historical health dataset (Pima Indians Dataset).
    """
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plot user's blood glucose vs population's blood glucose distribution
    axs[0, 0].hist(diabetes_data['Glucose 🩸'], bins=30, alpha=0.7, label="Population")
    axs[0, 0].axvline(user_data['Glucose 🩸'], color='red', linestyle='dashed', linewidth=2, label="You")
    axs[0, 0].set_title("🩸 Blood Glucose Comparison")
    axs[0, 0].set_xlabel("Glucose Levels")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].legend()

    # Plot user's BMI vs population's BMI distribution
    axs[0, 1].hist(diabetes_data['BMI ⚖️'], bins=30, alpha=0.7, label="Population")
    axs[0, 1].axvline(user_data['BMI ⚖️'], color='red', linestyle='dashed', linewidth=2, label="You")
    axs[0, 1].set_title("⚖️ BMI Comparison")
    axs[0, 1].set_xlabel("BMI")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].legend()

    # Plot user's age vs population age distribution
    axs[1, 0].hist(diabetes_data['Age 👶'], bins=30, alpha=0.7, label="Population")
    axs[1, 0].axvline(user_data['Age 👶'], color='red', linestyle='dashed', linewidth=2, label="You")
    axs[1, 0].set_title("👶 Age Comparison")
    axs[1, 0].set_xlabel("Age")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].legend()

    # Plot user's blood pressure comparison vs population
    axs[1, 1].hist(diabetes_data['BloodPressure 💓'], bins=30, alpha=0.7, label="Population")
    axs[1, 1].axvline(user_data['BloodPressure 💓'], color='red', linestyle='dashed', linewidth=2, label="You")
    axs[1, 1].set_title("💓 Blood Pressure Comparison")
    axs[1, 1].set_xlabel("Blood Pressure")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].legend()

    # Show plots
    plt.tight_layout()
    st.pyplot(fig)


# Streamlit User Interface
st.title("👩‍⚕️ Dr. Diabetic Risk Predictor 🩺")
st.write(""" 
Welcome to **Dr. Diabetic Risk Predictor**. Predict your risk of diabetes and get personalized insights! 🩺
""")

# Input health stats using Streamlit UI
st.subheader("📊 Enter your health information:")
user_data = {}
for feature in columns[:-1]:  # exclude the Outcome column
    user_value = st.number_input(f"Enter your {feature}:", min_value=0, value=0, step=1)
    user_data[feature] = user_value

# Preprocess user data
user_input_df = pd.DataFrame([user_data])
user_input_scaled = scaler.transform(user_input_df)

# Prediction button
if st.button("🔮 Predict My Risk 🏥"):
    user_prediction = model.predict(user_input_scaled)[0]
    st.subheader("🔎 Your Risk Prediction 💬")
    st.write(give_suggestion(user_prediction, user_data))
    
    st.subheader("📊 Your Health vs Normal Population")
    plot_patient_health(user_data, diabetes_data)
    st.info(f"✅ Model Accuracy: {accuracy:.2f}")
else:
    st.warning("🖐️ Input your health details and press the button to check your risk!")
