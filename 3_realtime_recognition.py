import cv2
import numpy as np
import joblib
import pyttsx3
import threading
from collections import deque
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_draw
from utils import extract_features

# ── Config ─────────────────────────────────────────────
MODEL_FILE      = "models/asl_model.pkl"
BUFFER_SIZE     = 15   # smoothing buffer — avoids flickery predictions
CONFIDENCE_MIN  = 0.7  # only show prediction if confidence is above this
HOLD_FRAMES     = 20   # frames to hold a sign before confirming
# ───────────────────────────────────────────────────────

print("Loading model...")
model = joblib.load(MODEL_FILE)
engine = pyttsx3.init()
print("Ready!")

# smoothing buffer - holds last N predictions
prediction_buffer = deque(maxlen=BUFFER_SIZE)
current_word      = []      # letters collected so far
sentence          = []      # completed words
hold_counter      = 0       # how long current sign has been held
last_added        = None    # prevents same letter being added twice

def speak(text):
    """Speak in a separate thread so it doesn't freeze the camera."""
    def _spead():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

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
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            raw_pred   = model.classes_[np.argmax(proba)]

            # add to smoothing buffer
            prediction_buffer.append(raw_pred)

            # only confirm prediction if buffer agrees majority
            if len(prediction_buffer) == BUFFER_SIZE:
                most_common = max(set(prediction_buffer),
                                  key=prediction_buffer.count)
                count = prediction_buffer.count(most_common)
                if count >= BUFFER_SIZE * 0.6 and confidence >= CONFIDENCE_MIN:
                    prediction = most_common
            # hold to confirm letter
            if prediction and prediction == last_added:
                hold_counter = 0 # reset if same letter already added
            elif prediction:
                hold_counter += 1
                if hold_counter >= HOLD_FRAMES:
                    current_word.append(prediction)
                    last_added   = prediction
                    hold_counter = 0
                    prediction_buffer.clear()

        else:
            hold_counter = 0
            last_added   = None


        # build display strings
        wword_str     = "".join(current_word)
        sentence_str = " ".join(sentence) + (" " + word_str if word_str else "")

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
            

            # hold progress bar
            progress = int((hold_counter / HOLD_FRAMES) * (w - 40))
            cv2.rectangle(frame, (20, 72), (20 + progress, 78),
                          (0, 200, 255), -1)
        
        else:
            cv2.putText(frame, "Show your hand...", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
            
        # Middle Bar - Current word being built
        cv2.rectangle(frame, (0, h - 110), (w, h - 70), (30, 30, 30), -1)
        cv2.putText(frame, f"{sentence_str}", (15, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, 
                    "SPACE=word  ENTER=speak  BKSP=delete  C=clear  Q=quit",
                    (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        cv2.imshow("ASL Recognition", frame)

        # KEYBOARD CONTROLS
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if current_word:
                sentence.append(wword_str)
                current_word = []
                last_added   = None
        elif key == 8:
            if current_word:
                current_word.pop()
        elif key == ord('c'):
            current_word = []
            sentence     = []
            last_added   = None

cap.release()
cv2.destroyAllWindows()