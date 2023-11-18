import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import torch
import numpy as np
import gdown
from models import Net
from utils import process_frame
import os
import av

# Function to download the model checkpoint
def download_checkpoint(url, output):
    if not os.path.exists(output):
        print("Downloading model checkpoint...")
        gdown.download(url, output, quiet=False)
    else:
        print("Model checkpoint already exists.")

# URL and output path for the model checkpoint
checkpoint_url = 'https://drive.google.com/uc?id=1oEPAduxUMG0j0hLWv4KlBehaxrtrz9Sr'
checkpoint_path = 'keypoints_model_1.pt'

# Download the model checkpoint only if it's not available
download_checkpoint(checkpoint_url, checkpoint_path)

# Load the model
net = Net()
net.load_state_dict(torch.load(checkpoint_path))
net.eval()  # Set the model to evaluation mode

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Custom Video Processor
class VideoProcessor(VideoTransformerBase):
    frame_skip = 15  # Process every nth frame to reduce load

    def __init__(self) -> None:
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return frame  # Skip processing for this frame

        image = frame.to_ndarray(format="bgr24")
        processed_image = process_frame(image, net, face_cascade)
        return av.VideoFrame.from_ndarray(processed_image, format="bgr24")

# Streamlit App Configuration
st.title("Real-time Emotion Detection")
st.write("This app detects your emotions in real-time using your webcam.")

# WebRTC streamer configuration
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Streamlit WebRTC component
webrtc_streamer(key="emotion_detection", 
                video_processor_factory=VideoProcessor,
                rtc_configuration=RTC_CONFIGURATION)
