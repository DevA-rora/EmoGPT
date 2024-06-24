# Import the necessary libraries:
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model # type: ignore (code works just fine, don't know why it's throwing an error)
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore (code works just fine, don't know why it's throwing an error)

# Load the pre-trained model and cascade classifier:
