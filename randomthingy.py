import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np

st.title("DeepFace and Streamlit Integration")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert the image to numpy array
    img_array = np.array(image)

    try:
        # Analyze the image
        result = DeepFace.analyze(img_array, actions=['age', 'gender', 'race', 'emotion'])
        st.write(result)
    except Exception as e:
        st.write("Error:", e)
