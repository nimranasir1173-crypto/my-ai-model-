import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Skin Cancer Detection")
st.write("Upload an image to classify it.")

# Model file
MODEL_PATH = Path(__file__).parent / "skin_cancer_model.keras"

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )
    return model

# Model loading
if not MODEL_PATH.exists():
    st.error("Model file not found! Check your project folder.")
    st.stop()

try:
    model = load_model()
    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Get input size
try:
    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    height = shape[1] or 224
    width = shape[2] or 224

except Exception as e:
    st.error(f"Input shape error: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # Preprocessing
        image = image.resize((width, height))

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = image_array / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Prediction
        if st.button("Detect Skin Cancer"):

            with st.spinner("Processing image..."):

                prediction = model.predict(
                    image_array,
                    verbose=0
                )

            prediction = np.asarray(prediction)

            if prediction.size == 0:
                st.error("Model returned an empty prediction.")

            elif prediction.ndim == 2 and prediction.shape[-1] == 1:

                score = float(prediction[0][0])

                if not np.isfinite(score):
                    st.error("Invalid prediction returned by model.")

                elif 0 <= score <= 1:
                    result = "Malignant" if score >= 0.5 else "Benign"
                    st.subheader(f"Result: {result}")
                    st.write(f"Model score: {score:.4f}")

                else:
                    st.error(
                        "Output is outside the expected probability range. "
                        "Check the model's final activation."
                    )

            elif prediction.ndim == 2:

                class_index = int(
                    np.argmax(prediction[0])
                )

                st.subheader(
                    f"Predicted Class: {class_index}"
                )

                st.write(
                    f"Model score: {float(np.max(prediction[0])):.4f}"
                )

            else:
                st.error(
                    f"Unexpected prediction shape: {prediction.shape}"
                )

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.caption(
    "Educational project only. Not intended for medical diagnosis."
)
