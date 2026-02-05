import streamlit as st
import joblib
import numpy as np

model = joblib.load("xgb_model.joblib")

st.title("Diabetes Prediction")

preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
bmi = st.number_input("BMI")
age = st.number_input("Age")

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, bmi, age]])
    pred = model.predict(input_data)

    if pred[0] == 1:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk")
