import cv2
import csv
import os
import time
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_draw
from utils import extract_features

# ── Config ─────────────────────────────────────────────────
DATA_FILE = "data/az_data.csv"
SAMPLES_PER_LABEL = 200         # how many frames to record per sign
COUNTDOWN = 3                   # seconds to get ready before recording starts
# ────────────────────────────────────────────────────────────

os.makedirs("data", exist_ok=True)

# ask what label to collect
label = input("Enter the sign label to collect (e.g. A, B, HELLO): ").strip().upper()
print(f"\nGet ready to sign '{label}' - recording {SAMPLES_PER_LABEL} samples.")
print("Press SPACE to start. Press Q to quit anytime.\n")

cap = cv2.VideoCapture(0)
collected = 0
recording = False
countdown_start = None

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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # ── draw landmarks ──────────────────────────────────
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lm,
                    mp_hands_module.HAND_CONNECTIONS
                )
        # ── countdown before recording ──────────────────────
        if countdown_start is not None:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN - int(elapsed)
            if remaining > 0:
                cv2.putText(frame, f"Starting in {remaining}...", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 3)
                
            else:
                recording = True
                countdown_start = None

        # ── record samples ──────────────────────────────────
        if recording and results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            features = extract_features(hand_lm)

            with open(DATA_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(features + [label])

            collected += 1

            #progress bar
            progress = int((collected / SAMPLES_PER_LABEL) *400)
            cv2.rectangle(frame, (20, 420), (420, 445), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 420), (20 + progress, 445), (0, 220, 100), -1)
            cv.putText(frame, f"Collecting '{label}': {collected}/{SAMPLES_PER_LABEL}",
                       (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 100), 2)
            
            if collected >= SAMPLES_PER_LABEL:
                print(f"Done! {SAMPLES_PER_LABEL} samples saved for '{label}'.")
                break

        elif not recording and countdown_start is None:
            cv2.putText(frame, f"Sign: {label}  |  SPACE = start  Q = quit",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
        cv2.imshow("ASL Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not recording:
            countdown_start = time.time()

cap.release()
cv2.destroyAllWindows()
print(f"\nData save to: {DATA_FILE}")


# it works intendedly :) -----> captures the gestures