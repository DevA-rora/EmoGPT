# Import required libraries:
import cv2
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import time

def main():
    st.title("Emotion Detection with Webcam and DeepFace!")

    # Page text:
    st.write("This app uses your webcam to detect your emotions in real-time.")
    st.write("Then, you will be able to see the detected emotion after you click the stop button.")
    st.write("After you get all your emotion stats, you will then be able to talk to ChatGPT. It will have the knowledge of your emotions and it will give responses accordingly.")

    # Initialize session state variables
    if 'snapshot_taken' not in st.session_state:
        st.session_state.snapshot_taken = False
    if 'photo_analyzed' not in st.session_state:
        st.session_state.photo_analyzed = False

    # Take snapshot button:
    if not st.session_state.snapshot_taken:
        if st.button("Take Snapshot"):
            take_photo()
            st.session_state.snapshot_taken = True

    # Analyze photo button:
    if st.session_state.snapshot_taken and not st.session_state.photo_analyzed:
        if st.button("Analyze Photo"):
            analyze_photo("person_snapshot.png")
            st.session_state.photo_analyzed = True

def take_photo():
    # Initialize webcam capture:
    cam = cv2.VideoCapture(0)
    
    # If the camera is not opened, display an error:
    if not cam.isOpened():
        st.error("Unable to access the camera")
        return

    # Placeholder for displaying the webcam feed
    stframe = st.empty()

    # Begin the while true loop:
    while True:
        # Sleep for 0.1 seconds so that cv2 can catch up:
        time.sleep(0.1)

        # Read the frame from the webcam
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        # Convert the frame to RGB format for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        # Sleep for 0.1 seconds so that cv2 can catch up:
        time.sleep(0.1)

        # Initialize the variable to store the photo that was taken:
        img_path = "person_snapshot.png"

        # Write the image to the storage
        cv2.imwrite(img_path, frame)

        st.success(f"""
            Success! \n
            {img_path} was captured!
        """)
        break

def analyze_photo(photo):
    # Load the image
    img = cv2.imread(photo)

    # Detect the face from the image
    try:
        face = DeepFace.detectFace(img,
            target_size = (224, 224,),
            detector_backend = "opencv"
        )

        # Display the face using Streamlit's st.image
        st.image(face, channels="RGB", caption="Detected Face")
    except Exception as e:
        st.error(f"Error in analyzing photo: {str(e)}")

if __name__ == "__main__":
    main()
