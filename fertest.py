import cv2
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

# Load pre-trained model and cascade classifier
@st.cache_resource
def load_resources():
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_model = load_model('cnn_emotion_detection.h5')  # You'll need to provide this model file
    return face_classifier, emotion_model

face_classifier, emotion_model = load_resources()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@st.cache_resource
def init_camera():
    return cv2.VideoCapture(0)

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_model.predict(roi)[0]
            emotion = emotion_labels[prediction.argmax()]
            return emotion

    return None

def main():
    st.title("Emotion Detection with Webcam")
    st.write("This app uses your webcam to detect your emotions.")

    camera = init_camera()

    if 'snapshot' not in st.session_state:
        st.session_state.snapshot = None

    frame_placeholder = st.empty()

    if st.button("Take Snapshot"):
        st.session_state.snapshot = take_photo(camera, frame_placeholder)

    if st.session_state.snapshot is not None:
        st.image(st.session_state.snapshot, channels="RGB", caption="Captured Snapshot")
        if st.button("Analyze Photo"):
            emotion = detect_emotion(st.session_state.snapshot)
            if emotion:
                st.write(f"Detected Emotion: {emotion}")
            else:
                st.write("No face detected in the image.")

def take_photo(camera, frame_placeholder):
    for _ in range(10):  # Try a few times to get a good frame
        ret, frame = camera.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", caption="Live Feed")
            
    if ret:
        st.success("Snapshot taken successfully!")
        return frame_rgb
    else:
        st.error("Failed to capture image. Please try again.")
        return None

if __name__ == "__main__":
    main()