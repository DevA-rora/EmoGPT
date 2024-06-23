import streamlit as st
import cv2
from PIL import Image
import numpy as np
from deepface import DeepFace

# Function to capture video frames from webcam
def capture_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access the camera.")
        return None
    
    while cap.isOpened() and not st.session_state.stop_recording:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame

# Function to detect emotions using DeepFace
def detect_emotions(image):
    try:
        result = DeepFace.analyze(image, actions=['emotion'])
        return result['dominant_emotion']
    except Exception as e:
        st.error(f"Error in emotion detection: {str(e)}")
        return None

# Streamlit app
def main():
    st.title("Emotion Detection with Webcam and DeepFace")
    
    # Initialize session state variables
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'stop_recording' not in st.session_state:
        st.session_state.stop_recording = False
    
    # Sidebar controls
    st.sidebar.markdown("# Controls")
    if not st.session_state.recording:
        if st.sidebar.button("Start Recording"):
            st.session_state.recording = True
            st.session_state.stop_recording = False
    else:
        if st.sidebar.button("Stop Recording"):
            st.session_state.stop_recording = True
    
    # Placeholder for displaying the webcam feed
    image_placeholder = st.empty()
    
    # Placeholder for displaying detected emotion
    emotion_placeholder = st.empty()
    
    # Start capturing and displaying webcam feed
    if st.session_state.recording:
        video_generator = capture_video()
        frame = next(video_generator)
        image_placeholder.image(frame, channels="RGB", use_column_width=True)
        
        for frame in video_generator:
            image_placeholder.image(frame, channels="RGB", use_column_width=True)
            if st.session_state.stop_recording:
                break
    
    # When recording stops, capture a screenshot and detect emotions
    if st.session_state.stop_recording:
        st.write("Recording stopped.")
        
        # Convert last frame to Image and display
        img = Image.fromarray(frame)
        image_placeholder.image(img, channels="RGB", use_column_width=True, caption="Last Frame")
        
        # Save the last frame as an image file
        img_path = "last_frame.png"
        cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Detect emotions using DeepFace
        st.write("Detecting emotions...")
        detected_emotion = detect_emotions(img_path)
        
        # Display the detected emotion
        if detected_emotion:
            emotion_placeholder.write(f"Detected emotion: {detected_emotion}")
        else:
            emotion_placeholder.error("Failed to detect emotion.")
    
if __name__ == "__main__":
    main()