# Import required libraries:
import cv2
import streamlit as st
from PIL import Image
import numpy as np
from deepface import DeepFace

def main():
    st.title("Emotion Detection with Webcam and DeepFace!")

    # Page text:
    st.write("This app uses your webcam to detect your emotions in real-time.")
    st.write("Then, you will be able to see the detected emotion after you click the stop button.")
    st.write("After you get all your emotion stats, you will then be able to talk to ChatGPT. It will have the knowledge of your emotions and it will give responses accordingly.")

    # Initialize session state variables
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'stop_recording' not in st.session_state:
        st.session_state.stop_recording = False

    # Sidebar controls
    # st.sidebar.markdown("# Controls")

    take_snapshot = st.button("Take Snapshot")
    if take_snapshot:
        take_photo()

def take_photo():
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        st.error("Unable to access the camera")
        return

    img_counter = 0
    stframe = st.empty()

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        # Convert the frame to RGB format for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        # Capture the frame on SPACE key press
        if st.button("Capture Image"):
            img_name = f"opencv_frame_{img_counter}.png"
            cv2.imwrite(img_name, frame)
            st.success(f"{img_name} written!")
            img_counter += 1
            break

    # Release the camera
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()