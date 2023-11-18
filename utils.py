# utils.py
import cv2
import numpy as np
import torch

def show_all_keypoints(image, keypoints, emotion, bbox):
    """
    Draws keypoints, a bounding box, and displays the detected emotion on the image.
    """
    # Draw keypoints
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)

    # Draw bounding box
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Put emotion text
    cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return image

def detect_emotion(keypoints):
    """
    Detect emotion based on keypoints.
    """
    # Hypothetical average distances in a neutral expression
    avg_mouth_distance = 10  # Average distance between upper and lower lip
    avg_eyebrow_distance = 15  # Average distance from eyebrow to reference point
    avg_eye_opening = 8  # Average distance between upper and lower eyelids

    # Setting thresholds as a percentage of these averages
    mouth_closed_threshold = avg_mouth_distance * 0.2
    eyebrow_raised_threshold = avg_eyebrow_distance * 1.2
    eye_opening_threshold = avg_eye_opening * 1.3

    # Check for smiling
    mouth_corners_up = keypoints[48][1] < keypoints[57][1] and keypoints[54][1] < keypoints[57][1]
    mouth_closed = abs(keypoints[51][1] - keypoints[57][1]) < mouth_closed_threshold

    # Check for frowning
    eyebrows_lower_than_reference = keypoints[21][1] > keypoints[27][1] and keypoints[22][1] > keypoints[27][1]

    # Check for surprise or fear
    eyebrows_raised = keypoints[21][1] < avg_eyebrow_distance - eyebrow_raised_threshold and keypoints[22][1] < avg_eyebrow_distance - eyebrow_raised_threshold
    eyes_wide_open = (abs(keypoints[43][1] - keypoints[47][1]) > eye_opening_threshold) and (abs(keypoints[37][1] - keypoints[41][1]) > eye_opening_threshold)

    # Logic for determining emotions
    if mouth_corners_up and mouth_closed:
        return "Smiling"
    elif eyebrows_lower_than_reference:
        return "Frowning"
    elif eyebrows_raised or eyes_wide_open:
        return "Surprised or Afraid"
    else:
        return "Neutral"

def process_frame(frame, net, face_cascade):
    """
    Processes each frame from the video stream.
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) with padding
        pad = 10
        roi = gray_frame[y-pad:y+h+pad, x-pad:x+w+pad]
        
        # Resize, normalize, and reshape the ROI
        roi_resize = cv2.resize(roi, (224, 224))
        roi_normalized = roi_resize / 255.0
        roi_reshaped = roi_normalized.reshape(1, 224, 224)

        # Convert the processed ROI to a tensor
        roi_tensor = torch.from_numpy(roi_reshaped).unsqueeze(0).float()

        # Predict keypoints
        with torch.no_grad():
            output_pts = net(roi_tensor)
            output_pts = output_pts.view(output_pts.size()[0], 68, -1)
            predicted_key_pts = (output_pts.numpy()[0] * 50.0 + 100)

        # Adjust keypoints to fit the original image scale
        keypoints = predicted_key_pts * (w / 224, h / 224) + (x, y)

        # Detect emotion
        emotion = detect_emotion(predicted_key_pts)

        # Draw keypoints, bounding box, and emotion on the frame
        frame = show_all_keypoints(frame, keypoints, emotion, (x, y, w, h))

    return frame
