# 🇵🇰 Pakistani Currency Recognition System Using Deep Learning

This repository contains an end-to-end Deep Learning web application designed to identify and classify Pakistani currency banknotes (**Rs. 10, 20, 50, 100, 500, 1000, and 5000**). 

The system doesn't just recognize the value, but also detects the spatial orientation (**Front or Back side**) of the banknote across **14 unique target classes**.

## 🧠 Model & Performance
* **Architecture:** MobileNetV2 (Fine-tuned via Transfer Learning)
* **Dataset Split:** 70% Training, 30% Testing/Validation
* **Validation Accuracy:** **88.42%**
* **Core Metrics Evaluated:** Precision, Recall, and F1-Score

## 🛠️ Tech Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Web Interface:** Streamlit
* **Development Environment:** Google Colab (NVIDIA T4 GPU) & VS Code
* **Image Processing:** Pillow (PIL) & NumPy

## 💻 How to Run Locally
1. Clone the repo and navigate to the folder.
2. Install dependencies: `pip install streamlit tensorflow pillow numpy`
3. Place your trained model file `pkr_final_model.h5` in the root directory.
4. Run the app: `streamlit run app.py`
