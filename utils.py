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

def list_available_camera(max_check=5, skip_index=None):
    available = []
    if skip_index is not None:
        available.append(skip_index)

    for i in range(max_check):
        if i == skip_index:
            continue
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(1)
            cap.release()
            
    return available if available else [0]

def compute_motion_features(landmark_buffer):
    """
    Given a buffer of landmark frames (list of 63-value feature vectors),
    compute motion characteristics: total displacement, direction, and shape.
    """
    if len(landmark_buffer) < 2:
        return None

    frames = np.array(landmark_buffer)

    fingertip_positions = frames[:, 24:26]

    diffs = np.diff(fingertip_positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    total_motion = np.sum(distances)

    start = fingertip_positions[0]
    end = fingertip_positions[-1]
    net_displacement = end - start
    direction_angle = np.degrees(np.arctan2(net_displacement[1], net_displacement[0]))

    return {
        "total_motion": total_motion,
        "direction_angle": direction_angle,
        "net_displacement": np.linalg.norm(net_displacement),
    }

def classify_dynamic_letter(motion_features, motion_threshold=0.15):
    """
    Rule-based classifier for dynamic letters based on motion direction.
    Thresholds are placeholders, need calibration against real AzSL motions.
    """
    if motion_features is None:
        return None

    total_motion = motion_features["total_motion"]
    angle = motion_features["direction_angle"]

    if total_motion < motion_threshold:
        return None

    if -30 <= angle <= 30:
        return "İ"
    elif 60 <= angle <= 120:
        return "Ü"
    elif 150 <= angle or angle <= -150:
        return "Ç"
    elif -120 <= angle <= -60:
        return "Ö"
    else:
        return None
