import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("cat_dog_classifier.h5")

# Title
st.title("Cat vs Dog Classifier 🐱🐶")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Show image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (256,256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    # Display result
    if prediction[0][0] > 0.5:
        st.success("Prediction: Dog 🐶")
    else:
        st.success("Prediction: Cat 🐱")
