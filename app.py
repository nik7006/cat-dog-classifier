import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# -----------------------------
# Load model safely (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cat_dog_classifier.h5")
    return model

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image and the model will predict whether it is a cat or dog.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = np.array(image)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    # Result
    if prediction[0][0] > 0.5:
        st.success(f"Prediction: Dog 🐶 ({prediction[0][0]*100:.2f}% confidence)")
    else:
        st.success(f"Prediction: Cat 🐱 ({(1-prediction[0][0])*100:.2f}% confidence)")
