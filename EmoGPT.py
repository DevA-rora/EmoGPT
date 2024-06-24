# Import the nessecary libraries:
from openai import OpenAI
import cv2
import matplotlib as plt
from deepface import DeepFace
from dotenv import load_dotenv, find_dotenv
import os

# Load the environment variables:
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Starting the Emotion detection analysis:

# Import detection models:

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

# Load the ChatGPT client with the API key:
client = OpenAI(api_key=os.getenv("API_KEY"))

# Initialize the list of messages with a user message:
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant who will be given the ability to recognize the emotions of the user. Your job is to make sure that your responses are tailored to the user's emotion. An example is if the user asks something such as: I can't play the piano, what should I do? give a response tailored to the emotion the user is experiencing. At the end of your response, give the emotion that you detected from the user."
    },
]

# Begin the text chain with ChatGPT:
while True:
    new_message = input("You: ").strip()
    
    if new_message == "exit":
        break

    # Optionally, determine the dominant emotion of the user
    new_message += f" (User is feeling {dominant_emotion})"

    # Append the new message to the list of messages:
    messages.append({
        "role": "user",
        "content": new_message
    })

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=messages)

    # Extract the assistant's message from the response:
    assistant_message = response.choices[0].message.content

    # Print the assistant's message:
    print("Assistant:", assistant_message)

    # Append the assistant's message to the list of messages:
    messages.append({
        "role": "assistant",
        "content": assistant_message
    })

print("Conversation ended.")