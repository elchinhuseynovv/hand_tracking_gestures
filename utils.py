import numpy as np

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