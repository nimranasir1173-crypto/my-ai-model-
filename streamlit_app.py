import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Page configuration
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Skin Cancer Detection")
st.write("Upload a skin image for educational classification.")

# Model path
MODEL_PATH = "skin_cancer_model.keras"

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)

# Load model safely
try:
    model = load_model()
    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Get model input dimensions
try:
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    h = int(input_shape[1]) if input_shape[1] else 224
    w = int(input_shape[2]) if input_shape[2] else 224

except Exception as e:
    st.error(f"Input shape error: {e}")
    st.stop()

# Image uploader
file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:

    try:
        # Read image
        img = Image.open(file).convert("RGB")

        st.image(
            img,
            caption="Uploaded Skin Image",
            use_container_width=True
        )

        # Resize image
        img = img.resize((w, h))

        # Convert image into array
        x = np.array(img, dtype=np.float32)

        # Normalize image
        x = x / 255.0

        # Add batch dimension
        x = np.expand_dims(x, axis=0)

        # Prediction
        with st.spinner("Analyzing image..."):
            pred = model.predict(x, verbose=0)

        pred = np.asarray(pred)

        if pred.ndim != 2 or pred.shape[0] != 1:
            st.error("Unexpected model output shape.")
            st.stop()

        st.write("Raw Prediction:", pred.tolist())

        # Classification
        if pred.shape[-1] == 1:

            score = float(pred[0][0])

            if score >= 0.5:
                label = "Malignant"
            else:
                label = "Benign"

            st.subheader(f"Result: {label}")
            st.write(f"Model score: {score:.4f}")

        else:
            class_index = int(np.argmax(pred[0]))
            st.subheader(f"Predicted Class: {class_index}")
            st.write(
                f"Confidence score: {float(np.max(pred[0])):.4f}"
            )

    except Exception as e:
        st.error(f"Image prediction error: {e}")

st.caption(
    "Educational project only. This application is not "
    "a medical diagnostic tool."
)
