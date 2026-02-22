import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("pneumonia_model2.h5")

st.title("Pneumonia Detector")
st.write("Upload a chest X-ray image to detect pneumonia")

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150,150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    result = "PNEUMONIA" if prediction > 0.5 else "NORMAL"

    st.image(uploaded_file, caption="Uploaded X-ray")
    st.subheader("Prediction:")
    st.success(result)
