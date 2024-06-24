import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
import os

# Load pre-trained models
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_model.h5')  # You'll need to train or download this model

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

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
    st.write("This app uses your webcam to detect your emotions in real-time.")

    if 'snapshot_taken' not in st.session_state:
        st.session_state.snapshot_taken = False
    if 'photo_analyzed' not in st.session_state:
        st.session_state.photo_analyzed = False

    if not st.session_state.snapshot_taken:
        if st.button("Take Snapshot"):
            take_photo()
            st.session_state.snapshot_taken = True

    if st.session_state.snapshot_taken and not st.session_state.photo_analyzed:
        if st.button("Analyze Photo"):
            analyze_photo("person_snapshot.png")
            st.session_state.photo_analyzed = True

def take_photo():
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        st.error("Unable to access the camera")
        return

    stframe = st.empty()
    
    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        img_path = "person_snapshot.png"
        cv2.imwrite(img_path, frame)

        if os.path.exists(img_path):
            st.success(f"Success! {img_path} was captured!")
            break
        else:
            st.error("Failed to save the image.")

    cam.release()

def analyze_photo(photo):
    img = cv2.imread(photo)
    if img is None:
        st.error("Could not load the image. Please check the file path.")
        return

    emotion = detect_emotion(img)

    if emotion:
        st.write(f"Detected Emotion: {emotion}")
    else:
        st.write("No face detected in the image.")

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Analyzed Image")

if __name__ == "__main__":
    main()