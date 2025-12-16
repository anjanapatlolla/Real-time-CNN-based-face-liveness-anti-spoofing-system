import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Streamlit setup
st.set_page_config(page_title="Real-Time Liveness Detection", layout="wide")
st.title("Real-Time CNN-Based Face Liveness & Anti-Spoofing System")

# Load the model once (cached)
@st.cache_resource
def load_liveness_model():
    return load_model("efficientnet_liveness_model.keras")

model = load_liveness_model()

# EAR calculation
def eye_aspect_ratio(landmarks, eye_indices):
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[3]]
    top = ((landmarks[eye_indices[1]][0] + landmarks[eye_indices[2]][0]) / 2,
           (landmarks[eye_indices[1]][1] + landmarks[eye_indices[2]][1]) / 2)
    bottom = ((landmarks[eye_indices[5]][0] + landmarks[eye_indices[4]][0]) / 2,
              (landmarks[eye_indices[5]][1] + landmarks[eye_indices[4]][1]) / 2)

    horiz_dist = np.linalg.norm(np.array(left) - np.array(right))
    vert_dist = np.linalg.norm(np.array(top) - np.array(bottom))
    return vert_dist / horiz_dist if horiz_dist != 0 else 0

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Webcam toggle
run = st.checkbox("Start Webcam")
frame_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        label = "Unknown"
        color = (0, 0, 0)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

                # EAR
                left_ear = eye_aspect_ratio(landmarks, [362, 385, 387, 263, 373, 380])
                right_ear = eye_aspect_ratio(landmarks, [33, 160, 158, 133, 153, 144])
                avg_ear = (left_ear + right_ear) / 2

                # CNN Prediction
                face_resized = cv2.resize(frame, (224, 224))
                face_array = img_to_array(face_resized) / 255.0
                face_array = np.expand_dims(face_array, axis=0)
                prediction = model.predict(face_array)[0]
                score = prediction[1]

                # Final label using AND logic
                if score > 0.5 and avg_ear > 0.25:
                    label = "Real"
                    color = (0, 255, 0)
                else:
                    label = "Fake"
                    color = (0, 0, 255)

                # Show only label
                cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        frame_placeholder.image(frame, channels="BGR")

    cap.release()
