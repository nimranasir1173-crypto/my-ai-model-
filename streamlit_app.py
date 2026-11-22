import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Skin Cancer Detection")
st.title("Skin Cancer Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_model.keras")

model = load_model()

# model ki input size automatically le lo
_, h, w, _ = model.input_shape

file = st.file_uploader("Image upload karo", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    x = np.array(img.resize((w, h))) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    st.write("Raw prediction:", pred.tolist())

    if pred.shape[-1] == 1:
        label = "Malignant" if pred[0][0] > 0.5 else "Benign"
    else:
        label = f"Class {int(np.argmax(pred))}"
    st.success(f"Result: {label}")

st.caption("Yeh sirf educational project hai, medical diagnosis nahi.")
