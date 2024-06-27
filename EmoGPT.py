# Import the necessary libraries:
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model # type: ignore (code works just fine, don't know why it's throwing an error)
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore (code works just fine, don't know why it's throwing an error)

# Load the pre-trained model and cascade classifier:

import os
import cv2
import numpy as np
from openai import OpenAI
import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Load the pre-trained model and cascade classifier
@st.cache_resource
def load_resources():
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    emotion_model = load_model("cnn_emotion_detection.h5")
    return face_classifier, emotion_model

face_classifier, emotion_model = load_resources()

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

@st.cache_resource
def init_camera():
    return cv2.VideoCapture(0)

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_model.predict(roi)[0]
            emotion = emotion_labels[prediction.argmax()]
            return emotion

    return None

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

def chatgpt_conversation(user_message, detected_emotion):
    # Initialize the OpenAI API client
    OpenAI_API_KEY = os.getenv("API_KEY")
    client = OpenAI(
        api_key = OpenAI_API_KEY
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant who can recognize the emotions of the user. Your job is to make sure that your responses are tailored to the user's emotion. An example is if the user asks something such as: I can't play the piano, what should I do? give a response tailored to the emotion the user is experiencing."
            }
        ]

    # Add the user's message and detected emotion to the conversation
    st.session_state.messages.append({
        "role": "user", 
        "content": f"{user_message} \n (Detected emotion: {detected_emotion})"
    })

    # Get the response from the API
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=st.session_state.messages)

    # Add the assistant's response to the conversation
    assistant_response = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    return assistant_response
def main():
    st.title("Emotion Detection and Chatbot")
    st.write("This app uses your webcam to detect your emotions and tailors chatbot responses accordingly.")

    camera = init_camera()

    if "step" not in st.session_state:
        st.session_state.step = 0

    if "snapshot" not in st.session_state:
        st.session_state.snapshot = None
    if "detected_emotion" not in st.session_state:
        st.session_state.detected_emotion = None

    frame_placeholder = st.empty()

    # Step 0: Take Snapshot
    if st.session_state.step == 0:
        if st.button("Take Snapshot"):
            st.session_state.snapshot = take_photo(camera, frame_placeholder)
            if st.session_state.snapshot is not None:
                st.session_state.step = 1

    # Step 1: Analyze Photo
    if st.session_state.step >= 1:
        st.image(st.session_state.snapshot, channels="RGB", caption="Captured Snapshot")
        if st.button("Analyze Photo"):
            st.session_state.detected_emotion = detect_emotion(st.session_state.snapshot)
            if st.session_state.detected_emotion:
                st.write(f"Detected Emotion: {st.session_state.detected_emotion}")
                st.session_state.step = 2
            else:
                st.write("No face detected in the image. Please try taking another snapshot.")
                st.session_state.step = 0

    # Step 2: Chat with the emotion-aware assistant
    if st.session_state.step == 2:
        st.write("Chat with the emotion-aware assistant:")
        user_message = st.text_input("You:", key="user_input")

        if st.button("Send"):
            if user_message:
                response = chatgpt_conversation(user_message, st.session_state.detected_emotion)
                st.text_area("Assistant:", value=response, height=100, key="assistant_response")
            else:
                st.warning("Please enter a message.")

        # Display conversation history
        st.write("Conversation History:")
        for message in st.session_state.get("messages", [])[1:]:  # Skip the system message
            st.text_area(f"{message['role'].capitalize()}:", value=message['content'], height=75, key=f"{message['role']}_{message['content'][:20]}")


if __name__ == "__main__":
    main()