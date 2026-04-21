import cv2
import numpy as np
import joblib
from collections import deque
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_draw
from utils import extract_features

# ── Config ─────────────────────────────────────────────
MODEL_FILE      = "models/asl_model.pkl"
BUFFER_SIZE     = 10   # smoothing buffer — avoids flickery predictions
CONFIDENCE_MIN  = 0.6  # only show prediction if confidence is above this
# ───────────────────────────────────────────────────────

print("Loading model...")
model = joblib.load(MODEL_FILE)
print("Model loaded! Starting webcam...")

# smoothing buffer - holds last N predictions
prediction_buffer = deque(maxlen=BUFFER_SIZE)

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

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BAYER_BGR2RGB)
        results = hands.process(rgb)

        prediction = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]

            # draw skeleton
            mp_draw.draw_landmarks(
                frame, hand_lm,
                mp_hands_module.HAND_CONNECTIONS
            )

            # extract features and predict
            features = extract_features(hand_lm)
            features_np = np.array(features).reshape(1, -1)

            proba      = model.predict_proba(features_np)[0]
            confidence = np.max(proba)
            rew_pred   = model.classes_[np.argmax(proba)]

            # add to smoothing buffer
            prediction_buffer.append(raw_pred)

            # only confirm prediction if buffer agrees majority
            if len(prediction_buffer) == BUFFER_SIZE:
                most_common = max(set(prediction_buffer),
                                  key=prediction_buffer.count)
                count = prediction_buffer.count(most_common)
                if count >= BUFFER_SIZE * 0.6 and confidence >= CONFIDENCE_MIN:
                    prediction = most_common

        # DRAW HUD
        h, w = frame.shape[:2]

        # background bar at top
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)

        if prediction:
            # big letter display
            cv2.putText(frame, prediction, (30, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 120), 4)
            
            # confidence bar
            bar_width = int((w - 200) * confidence)
            cv2.rectangle(frame, (150, 25), (w - 20, 55), (50, 50, 50), -1)
            cv2.rectangle(frame, (150, 25), (150 + bar_width, 55),
                          (0, 255, 120), -1)
            cv2.putText(frame, f"{confidence * 100:.0f}%", (155, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        
        else:
            cv2.putText(frame, "Show your hand...", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
            
        # controls hint at bottom
        cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, "Q = quit", (15, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        cv2.imshow("ASL Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
                        