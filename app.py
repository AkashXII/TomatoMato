import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image

from fusion import fuse_predictions

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Crop Health System",
    layout="centered"
)

st.title("TomatoMato—Tomato Crop Health Assessment System 🍅")
st.write("Upload a tomato leaf image and enter soil/environment details.")

# --------------------------------------------------
# Load models (load once)
# --------------------------------------------------
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model(
        "models/cnn_crop_disease_model.h5"
    )
    rf_model = joblib.load(
        "models/rf_crop_suitability.pkl"
    )
    return cnn_model, rf_model


cnn_model, rf_model = load_models()

# --------------------------------------------------
# Class names (ORDER MUST MATCH TRAINING)
# --------------------------------------------------
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# --------------------------------------------------
# Image upload
# --------------------------------------------------
uploaded_image = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Soil & environment inputs
# --------------------------------------------------
st.subheader("Soil & Environment Parameters")

N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=90)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=145, value=42)
K = st.number_input("Potassium (K)", min_value=0, max_value=205, value=43)

temperature = st.number_input(
    "Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0
)
humidity = st.number_input(
    "Humidity (%)", min_value=0.0, max_value=100.0, value=80.0
)
ph = st.number_input(
    "Soil pH", min_value=0.0, max_value=14.0, value=6.5
)
rainfall = st.number_input(
    "Rainfall (mm)", min_value=0.0, max_value=300.0, value=120.0
)

rf_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

# --------------------------------------------------
# Analyze button
# --------------------------------------------------
if st.button("Analyze Crop"):

    if uploaded_image is None:
        st.warning("Please upload a leaf image first.")

    else:
        # -----------------------------
        # Image preprocessing
        # -----------------------------
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Leaf", use_column_width=True)

        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # -----------------------------
        # CNN prediction
        # -----------------------------
        cnn_preds = cnn_model.predict(image_array)
        disease_index = int(np.argmax(cnn_preds))
        disease_confidence = float(np.max(cnn_preds))
        disease_name = CLASS_NAMES[disease_index]

        # -----------------------------
        # Random Forest prediction
        # -----------------------------
        rf_probs = rf_model.predict_proba(rf_input)
        suitability_score = float(np.max(rf_probs))

        # -----------------------------
        # Fusion
        # -----------------------------
        fusion_result = fuse_predictions(
            disease_confidence=disease_confidence,
            suitability_score=suitability_score
        )

        # -----------------------------
        # Display results
        # -----------------------------
        st.subheader("Results")

        st.write(f"**Detected Condition:** {disease_name}")
        st.write(f"**Disease Confidence:** {disease_confidence:.2f}")
        st.write(f"**Environmental Suitability Score:** {suitability_score:.2f}")

        st.markdown("---")

        st.write(f"### Overall Status: **{fusion_result['status']}**")
        st.write(f"**Final Score:** {fusion_result['final_score']}")
        st.write(f"**Recommendation:** {fusion_result['recommendation']}")
