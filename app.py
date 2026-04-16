import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(
    page_title="AI vs Real Detector",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .title { text-align: center; color: #00d4ff; font-size: 2.5rem; font-weight: bold; }
    .subtitle { text-align: center; color: #888; font-size: 1rem; margin-bottom: 2rem; }
    .result-real { background: #1a472a; padding: 1rem; border-radius: 10px; text-align: center; color: #51cf66; font-size: 1.5rem; }
    .result-fake { background: #4a1010; padding: 1rem; border-radius: 10px; text-align: center; color: #ff6b6b; font-size: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🔍 AI vs Real Image Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload any image — I will detect if it is Real or AI Generated!</p>', unsafe_allow_html=True)

st.divider()

model = tf.keras.models.load_model("ai_detector_model.h5")

uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🤖 Analysis Result")
        
        with st.spinner("Analyzing..."):
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            result = prediction[0][0]
        
        if result > 0.5:
            st.markdown('<div class="result-real">✅ REAL IMAGE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-fake">❌ AI GENERATED</div>', unsafe_allow_html=True)
        
        st.subheader("📊 Confidence Score")
        confidence = result if result > 0.5 else 1 - result
        st.progress(float(confidence))
        st.write(f"**{round(confidence * 100, 2)}% confident**")

st.divider()

with st.expander("ℹ️ About This Project"):
    st.write("""
    **AI vs Real Image Detector** is a deep learning project that detects 
    whether an image is real or AI generated.
    
    - **Model:** MobileNetV2 + CNN
    - **Dataset:** 1,00,000 images
    - **Accuracy:** 90%+
    - **Developer:** Neetesh | B.Tech CSE AI
    """)