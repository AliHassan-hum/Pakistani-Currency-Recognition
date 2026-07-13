import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os
from keras.models import load_model  # <-- TensorFlow ki jagah direct Keras import kiya

# 1. Page Configuration
st.set_page_config(page_title="PKR Currency Detector Pro", page_icon="🇵🇰", layout="centered")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #006600; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🇵🇰 Pakistani Currency Recognition")
st.write("System Status: **High Accuracy Mode (88.4%)**")
st.divider()

# 2. Model Loading
@st.cache_resource
def load_my_model():
    model_path = 'pkr_final_model.h5' 
    if os.path.exists(model_path):
        try:
            return load_model(model_path)  # <-- Yahan se 'tf.keras.models.' hata diya
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Error: {model_path} not found! Please check your folder.")
        return None

model = load_my_model()

# 3. Class Names (Sorted Alphabetically)
class_names = [
    '10_back', '10_front', '100_back', '100_front', 
    '1000_back', '1000_front', '20_back', '20_front', 
    '50_back', '50_front', '500_back', '500_front', 
    '5000_back', '5000_front'
]

# 4. Main Logic
if model is not None:
    uploaded_file = st.file_uploader("📤 Upload Currency Image (Front or Back)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load and convert image
        image = Image.open(uploaded_file)
        image = image.convert('RGB')
        
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with st.spinner('🔍 AI is analyzing security features...'):
            # Preprocessing
            size = (224, 224)
            image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image_resized) / 255.0
            img_reshape = img_array[np.newaxis, ...]

            # Prediction
            prediction = model.predict(img_reshape)
            result_index = np.argmax(prediction)
            result_label = class_names[result_index]
            confidence = np.max(prediction) * 100

        # 5. Display Results
        st.subheader("Final Result:")
        
        display_name = result_label.replace('_', ' ').upper()
        
        if confidence > 80:
            st.balloons()
            st.success(f"### Predicted Note: **{display_name}**")
        elif confidence > 60:
            st.warning(f"### Predicted Note: **{display_name}** (Moderate Confidence)")
        else:
            st.error(f"### Predicted Note: **UNCERTAIN**")
            st.write(f"AI Suggestion: {display_name}")

        st.info(f"**AI Confidence Score:** {confidence:.2f}%")
        
        if confidence < 70:
            st.write("💡 *Tip: Clear background aur behtar light mein dobara koshish karen.*")

st.divider()
st.caption("Developed for PKR Currency Project | Powered by MobileNetV2 Fine-Tuning")
