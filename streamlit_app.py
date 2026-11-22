import streamlit as st
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Skin Cancer Detection")
st.title("Skin Cancer Detection")
st.caption("App version: v3")

@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(
        "skin_cancer_model.keras", compile=False, safe_mode=False
    )

try:
    model = load_model()
except BaseException as e:
    st.error("Model could not be loaded. Error details:")
    st.code(f"{type(e).__name__}: {e}")
    st.stop()

file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if file is not None:
    try:
        img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
        st.image(img, caption="Uploaded image", width=350)
        x = np.array(img.resize((224, 224)), dtype="float32") / 255.0
        x = np.expand_dims(x, axis=0)
        p = float(model.predict(x, verbose=0).reshape(-1)[0])
        if p > 0.5:
            st.error(f"Malignant (signs of cancer): {p*100:.1f}%")
        else:
            st.success(f"Benign (no cancer): {(1-p)*100:.1f}%")
    except Exception as e:
        st.error("Prediction error:")
        st.code(f"{type(e).__name__}: {e}")

st.caption("This is an educational project only, not a medical diagnosis. Please consult a doctor for a proper check-up.")
