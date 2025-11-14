import streamlit as st
import pickle
import numpy as np

# -------------------------------
# RECOMMENDATION DATA
# -------------------------------
crop_info = {
    "apple": {
        "requirements": "Requires cool climate, pH 5.5–6.5, well‑drained loamy soil.",
        "growing_tips": "Plant in full sunlight, prune regularly, ensure proper irrigation, protect from pests.",
        "profit": "High market demand; profit ranges from ₹3–5 lakh per acre depending on variety."
    },
    "banana": {
        "requirements": "Hot, humid climate, pH 6.0–7.5, rich alluvial soil.",
        "growing_tips": "Maintain drip irrigation, apply organic manure, use proper spacing.",
        "profit": "₹2–3 lakh per acre with proper irrigation and maintenance."
    },
    "blackgram": {
        "requirements": "Warm climate, pH 6.0–7.0, loamy soil.",
        "growing_tips": "Needs good seed treatment, moderate irrigation, weed control.",
        "profit": "₹50,000–₹1.2 lakh per acre depending on season."
    },
    "chickpea": {
        "requirements": "Cool and dry climate, pH 6.0–9.0, loamy soil.",
        "growing_tips": "Avoid waterlogging, ensure proper spacing, apply phosphorus fertilizer.",
        "profit": "₹70,000–₹1.5 lakh per acre."
    },
    "coconut": {
        "requirements": "Tropical climate, pH 5.0–8.0, sandy to alluvial soil.",
        "growing_tips": "Deep watering, mulching, regular manure, pest management.",
        "profit": "₹2–6 lakh per acre yearly from nuts and byproducts."
    },
    "coffee": {
        "requirements": "Cool, shaded environment, pH 6.0–6.5, rich loamy soil.",
        "growing_tips": "Shade regulation, adequate rainfall, selective picking.",
        "profit": "₹1–3 lakh per acre depending on quality."
    },
    "cotton": {
        "requirements": "Warm climate, pH 5.8–8.0, black soil preferred.",
        "growing_tips": "Regular irrigation, pest control (especially bollworm).",
        "profit": "₹60,000–₹1.5 lakh per acre."
    },
    "grapes": {
        "requirements": "Hot and dry climate, pH 6.5–7.5, well‑drained soil.",
        "growing_tips": "Trellis system, pruning, proper irrigation.",
        "profit": "₹3–10 lakh per acre for table grapes."
    },
    "jute": {
        "requirements": "Warm, humid climate, pH 6.0–7.0, alluvial soil.",
        "growing_tips": "Needs good rainfall, proper retting process.",
        "profit": "₹40,000–₹1 lakh per acre."
    },
    "kidneybeans": {
        "requirements": "Moderate climate, pH 6.0–7.5.",
        "growing_tips": "Moderate irrigation, nitrogen fixing bacteria inoculation.",
        "profit": "₹60,000–₹1.2 lakh per acre."
    },
    "lentil": {
        "requirements": "Cool climate, pH 6.0–8.0.",
        "growing_tips": "Minimal irrigation, avoid waterlogging.",
        "profit": "₹50,000–₹1 lakh per acre."
    },
    "maize": {
        "requirements": "Warm climate, pH 5.5–7.0.",
        "growing_tips": "Requires good sunlight, nitrogen fertilizer.",
        "profit": "₹40,000–₹1.2 lakh per acre."
    },
    "mango": {
        "requirements": "Tropical climate, pH 5.5–7.5.",
        "growing_tips": "Proper pruning, yearly fertilization, irrigation.",
        "profit": "₹2–6 lakh per acre annually."
    },
    "mothbeans": {
        "requirements": "Hot and dry climate, pH 7.0–8.5.",
        "growing_tips": "Drought resistant; minimal irrigation.",
        "profit": "₹40,000–₹80,000 per acre."
    },
    "mungbean": {
        "requirements": "Warm climate, pH 6.2–7.2.",
        "growing_tips": "Regular weeding, moderate irrigation.",
        "profit": "₹40,000–₹1 lakh per acre."
    },
    "muskmelon": {
        "requirements": "Hot climate, pH 6.0–7.0.",
        "growing_tips": "Requires well‑drained soil, drip irrigation helpful.",
        "profit": "₹1–2 lakh per acre."
    },
    "orange": {
        "requirements": "Sub‑tropical climate, pH 5.0–6.5.",
        "growing_tips": "Regular watering, nutrient management, pest control.",
        "profit": "₹2–4 lakh per acre."
    },
    "papaya": {
        "requirements": "Warm climate, pH 6.0–6.5.",
        "growing_tips": "Need well‑drained soil, drip irrigation, pest control.",
        "profit": "₹3–5 lakh per acre."
    },
    "pigeonpeas": {
        "requirements": "Tropical climate, pH 6.0–7.0.",
        "growing_tips": "Deep sowing, moderate irrigation.",
        "profit": "₹60,000–₹1.5 lakh per acre."
    },
    "pomegranate": {
        "requirements": "Hot dry climate, pH 6.0–7.5.",
        "growing_tips": "Drip irrigation, pruning, disease control.",
        "profit": "₹3–8 lakh per acre."
    },
    "rice": {
        "requirements": "High rainfall, pH 5.0–6.5, clay soil.",
        "growing_tips": "Flood irrigation, proper spacing.",
        "profit": "₹40,000–₹80,000 per acre."
    },
    "watermelon": {
        "requirements": "Hot climate, pH 6.0–7.5.",
        "growing_tips": "Requires sandy soil, regular irrigation.",
        "profit": "₹1–3 lakh per acre."
    }
}

# -------------------------------
# LOAD MODEL
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))


# -------------------------------
# UI SETUP
# -------------------------------
st.set_page_config(page_title="AgriSmart Crop Recommender", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #2E7D32;'>🌾 AgriSmart – Smart Crop Recommendation System</h1>
    <p style='text-align: center;'>Your intelligent assistant for modern farming</p>
""", unsafe_allow_html=True)

st.subheader("📥 Enter Soil & Weather Values")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, 50)
    P = st.number_input("Phosphorus (P)", 0, 200, 50)
    K = st.number_input("Potassium (K)", 0, 200, 50)

with col2:
    temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)

with col3:
    ph = st.number_input("pH Level", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

if st.button("🌱 Predict Best Crop"):
    features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    
    prediction_index = model.predict(features)[0]
    prediction = label_encoder.inverse_transform([prediction_index])[0]


    st.success(f"Recommended Crop: **{prediction.capitalize()}**")

    if prediction in crop_info:
        data = crop_info[prediction]
        st.subheader("📗 Crop Insights")
        st.write(f"**Requirements:** {data['requirements']}")
        st.write(f"**Growing Tips:** {data['growing_tips']}")
        st.write(f"**Profit Estimate:** {data['profit']}")

st.markdown("---")
st.markdown("<p style='text-align:center;'>Developed with ❤️ for Smart Farming</p>", unsafe_allow_html=True)

