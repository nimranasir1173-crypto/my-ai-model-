import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

st.set_page_config(page_title="Skin Cancer Detection")
st.title("Skin Cancer Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_model.keras")

model = load_model()

file = st.file_uploader("Skin ki image upload karo", type=["jpg", "jpeg", "png"])

if file:
   img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    st.image(img, caption="Uploaded image", width=350)

    x = np.array(img.resize((224, 224))) / 255.0
    x = np.expand_dims(x, axis=0)

    p = float(model.predict(x)[0][0])
    if p > 0.5:
        st.error(f"Malignant (cancer ke asar): {p*100:.1f}%")
    else:
        st.success(f"Benign (cancer nahi): {(1-p)*100:.1f}%")

st.caption("This is an educational project only, not a medical diagnosis. Please consult a doctor for a proper check-up.")
