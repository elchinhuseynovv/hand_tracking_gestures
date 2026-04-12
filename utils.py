import numpy as np

def extract_features(hand_landmarks):
    """
    converts 21 MediaPipe landmarks into a normalized 63-element feature vector.
    all points are made relative to the wrist (landmark 0), then flattened.
    """

    landmarks = hand_landmarks.landmark
    wrist = landmarks[0]

    features = []
    for lm in landmarks:
        features.extend([
            lm.x - wrist.x,     # relative x
            lm.y - wrist.y,     # relative y
            lm.z - wrist.z,     # relative z
        ])

    scale = np.linalg.norm([
        landmarks[12].x - wrist.x,
        landmarks[12].y - wrist.y,
    ])
    if scale > 0:
        features = [f / scale for f in features]

    return features # 63 values