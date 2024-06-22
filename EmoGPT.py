# Import the nessecary libraries:
import openai
import cv2
import matplotlib as plt
from deepface import DeepFace
import os


# Starting the Emotion detection analysis:

# Import detection models:
emotion_model = DeepFace.build_model('Emotion') # Emotion detection model
gender_model =  DeepFace.build_model('Gender') # Gender detection model
race_model = DeepFace.build_model('Race') # Race detection model

# Initialise camera:
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from the camera
    ret, frame = video_capture.read()

    if not ret:
        # If no frame is received, continue to the next iteration
        continue

    # Perform face analysis
    result = DeepFace.analyze(frame, actions=['emotion', 'race', 'gender'], enforce_detection=False)

    # Get the predicted emotion, race, gender, and age
    emotion = result[0]['emotion']
    race = result[0]['race']
    gender = result[0]['gender']

    # Display the predicted results on the frame
    cv2.putText(frame, "Emotion: {}".format(emotion), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, "Race: {}".format(race), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, "Gender: {}".format(gender), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Get the Dominant Results and save to variable
    dominant_gender = max(gender, key=gender.get)
    dominant_emotion = max(emotion, key=emotion.get)
    dominant_race = max(race, key=race.get)

    # Display the dominant results on the frame:
    cv2.putText(frame, "Dominant Gender: {}".format(dominant_gender), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, "Dominant Emotion: {}".format(dominant_emotion), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, "Dominant Race: {}".format(dominant_race), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0,), 2)

    # Display the frame with the predicted results
    cv2.imshow('Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release video capture:
video_capture.release()
cv2.destroyAllWindows()


# Setup OpenAI ChatGPT Conversation:
