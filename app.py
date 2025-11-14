import streamlit as st
import joblib
import numpy as np

model = joblib.load("crop_model.pkl")

st.title("🌾 Crop Recommendation System")
st.write("Enter soil and climate values to predict the most suitable crop.")

# Input fields
N = st.number_input("Nitrogen (N)", 0, 200)
P = st.number_input("Phosphorus (P)", 0, 200)
K = st.number_input("Potassium (K)", 0, 200)
temp = st.number_input("Temperature (°C)", 0.0, 50.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0)
ph = st.number_input("Soil pH", 0.0, 14.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0)

if st.button("Predict Crop"):
    inputs = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    prediction = model.predict(inputs)[0]
    
    st.success(f"Recommended Crop: **{prediction.upper()}**")