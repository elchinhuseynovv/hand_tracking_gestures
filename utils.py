import numpy as np
import cv2

def extract_features(hand_landmarks):
    landmarks = hand_landmarks.landmark
    wrist = landmarks[0]

    features = []
    for lm in landmarks:
        features.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])

    scale = np.linalg.norm([
        landmarks[12].x - wrist.x,
        landmarks[12].y - wrist.y,
    ])
    if scale > 0:
        features = [f / scale for f in features]

    return features  # 63 values

def list_available_camera(max_check=5):
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(1)
            cap.release()
    return available if available else [0]
