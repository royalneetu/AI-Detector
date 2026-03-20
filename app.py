import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("ai_detector_model.h5")
st.title("AI vs Real Image Detector")
st.write("Image upload karo — main bataunga Real hai ya AI Generated!")
uploaded_file = st.file_uploader("Image choose karo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Tumhari image", width=300)
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    result = prediction[0][0]
    
    if result > 0.5:
        st.success("✅ Real Image hai!")
    else:
        st.error("❌ AI Generated Image hai!")
        
    st.write(f"Confidence: {round(result * 100, 2)}%")