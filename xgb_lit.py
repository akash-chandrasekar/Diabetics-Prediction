import streamlit as st
import numpy as np
from xgboost import XGBClassifier

# Load model
model = XGBClassifier()
model.load_model("diabetes_xgb.json")

st.title("🩺 Diabetes Prediction App")

st.write("Enter patient details below:")

# User Inputs
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 300)
bp = st.number_input("Blood Pressure", 0, 150)
bmi = st.number_input("BMI", 0.0, 70.0)
age = st.number_input("Age", 0, 100)

# Prediction Button
if st.button("Predict Diabetes"):

    input_data = np.array([[pregnancies, glucose, bp, bmi, age]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
