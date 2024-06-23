import streamlit as st
import cv2
from deepface import DeepFace
import threading

# Initialize session state for stopping the video capture
if 'stop' not in st.session_state:
    st.session_state.stop = True

# Load the DeepFace models
st.write("Loading models...")
emotion_model = DeepFace.build_model('Emotion')  # Emotion detection model
gender_model = DeepFace.build_model('Gender')    # Gender detection model
race_model = DeepFace.build_model('Race')        # Race detection model
st.write("Models loaded successfully!")

# Initialize camera
cap = cv2.VideoCapture(0)

st.title("Video capture with opencv")

frame_placeholder = st.empty()

start_button_pressed = st.button("Start video capture")
stop_button_pressed = st.button("Stop video capture")

def analyze_frame(frame):
    result = DeepFace.analyze(frame, actions=['emotion', 'race', 'gender'], enforce_detection=False)
    emotion = result[0]['emotion']
    race = result[0]['race']
    gender = result[0]['gender']

    # Get dominant results
    dominant_gender = max(gender, key=gender.get)
    dominant_emotion = max(emotion, key=emotion.get)
    dominant_race = max(race, key=race.get)

    return dominant_gender, dominant_emotion, dominant_race

def capture_frames():
    while cap.isOpened() and not st.session_state.stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Error reading frame")
            break

        dominant_gender, dominant_emotion, dominant_race = analyze_frame(frame)

        # Display the results on the Streamlit app
        frame_placeholder.image(frame, channels="BGR", caption=f"Emotion: {dominant_emotion}, Race: {dominant_race}, Gender: {dominant_gender}")

if start_button_pressed:
    st.session_state.stop = False
    threading.Thread(target=capture_frames).start()

if stop_button_pressed:
    st.session_state.stop = True
    cap.release()