# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import torch
import numpy as np
import gdown
from models import Net
from utils import process_frame

# Function to download the model checkpoint
def download_checkpoint(url, output):
    gdown.download(url, output, quiet=False)

# URL and output path for the model checkpoint
checkpoint_url = 'https://drive.google.com/uc?id=1oEPAduxUMG0j0hLWv4KlBehaxrtrz9Sr'
checkpoint_path = 'keypoints_model_1.pt'

# Download the model checkpoint
download_checkpoint(checkpoint_url, checkpoint_path)

# Load your model and any other necessary global variables
net = Net()
net.load_state_dict(torch.load('keypoints_model_1.pt'))
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        processed_image = process_frame(image, net, face_cascade)
        return av.VideoFrame.from_ndarray(processed_image, format="bgr24")

st.title("Real-time Emotion Detection")
st.write("This app detects your emotions in real-time using your webcam.")

webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
