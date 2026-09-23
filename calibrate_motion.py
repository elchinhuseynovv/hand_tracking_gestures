import cv2
from collections import deque
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_draw
from utils import extract_features, compute_motion_features

landmark_buffer = deque(maxlen=15)

cap = cv2.VideoCapture(0)

with mp_hands_module.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.6
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands_module.HAND_CONNECTIONS)

            features = extract_features(hand_lm)
            landmark_buffer.append(features)

            motion = compute_motion_features(list(landmark_buffer))
            if motion:
                text1 = f"Motion: {motion['total_motion']:.3f}"
                text2 = f"Angle: {motion['direction_angle']:.1f} deg"
                cv2.putText(frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, text2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, "Press Q to quit", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.imshow("Motion Calibration", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()