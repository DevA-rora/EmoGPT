import os
import cv2
import openai
import numpy as np
from dotenv import load_dotenv, search_dotenv

# I don't know why using tensorflow.keras is throwing an error, but I'll ignore it for now, especially because the two imports work as expected.
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore 

# Initialize the .env variables:
print("placeholder lol")

# Load the pre-trained model and cascade classifier:
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('cnn_emotion_detection.h5')  # Load the model for the emotion recognition
# According to the Kaggle source, this model is accurate roughly in the 80-90% range (based on what I think a "high accuracy" model means).

# Define the emotion labels:
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# Initialize the camera:
camera = cv2.VideoCapture(0)


# Start the conversation with ChatGPT: