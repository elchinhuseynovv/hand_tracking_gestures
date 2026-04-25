import cv2
import numpy as np
import joblib
import pyttsx3
import threading
from collections import deque
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions import drawing_styles as mp_styles
from utils import extract_features

# ── Config ─────────────────────────────────────────────
MODEL_FILE      = "models/az_model.pkl"
BUFFER_SIZE     = 10
CONFIDENCE_MIN  = 0.55
HOLD_FRAMES     = 15
HISTORY_SIZE    = 10
# ───────────────────────────────────────────────────────

# ── Colors (BGR) ───────────────────────────────────────
CLR_BG     = (15, 15, 15)
CLR_PANEL  = (30, 30, 30)
CLR_GREEN  = (0, 220, 100)
CLR_CYAN   = (200, 220, 0)
CLR_WHITE  = (240, 240, 240)
CLR_GRAY   = (120, 120, 120)
CLR_DARK   = (50, 50, 50)
CLR_ACCENT = (0, 165, 255)
# ───────────────────────────────────────────────────────

print("Loading model...")
model  = joblib.load(MODEL_FILE)
engine = pyttsx3.init()
engine.setProperty('rate', 150)
print("Ready!")

prediction_buffer = deque(maxlen=BUFFER_SIZE)
letter_history    = deque(maxlen=HISTORY_SIZE)
current_word      = []
sentence          = []
hold_counter      = 0
last_added        = None

def speak(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

def draw_rounded_rect(img, x1, y1, x2, y2, radius, color, thickness=-1):
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def draw_hud(frame, prediction, confidence, hold_counter,
             current_word, sentence, letter_history):
    h, w = frame.shape[:2]

    # ── Top panel ───────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), CLR_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    if prediction:
        draw_rounded_rect(frame, 15, 8, 90, 82, 8, CLR_PANEL)
        cv2.putText(frame, prediction, (25, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, CLR_GREEN, 4)

        cv2.putText(frame, "CONFIDENCE", (105, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_GRAY, 1)
        cv2.rectangle(frame, (105, 35), (w - 20, 58), CLR_DARK, -1)
        bar_w     = int((w - 125) * confidence)
        bar_color = CLR_GREEN if confidence > 0.75 else CLR_ACCENT
        cv2.rectangle(frame, (105, 35), (105 + bar_w, 58), bar_color, -1)
        cv2.putText(frame, f"{confidence*100:.0f}%", (110, 53),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_BG, 2)

        cv2.putText(frame, "HOLD", (105, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_GRAY, 1)
        cv2.rectangle(frame, (145, 64), (w - 20, 78), CLR_DARK, -1)
        hold_w = int((w - 165) * (hold_counter / HOLD_FRAMES))
        cv2.rectangle(frame, (145, 64), (145 + hold_w, 78), CLR_CYAN, -1)

    else:
        cv2.putText(frame, "Show your hand to begin...", (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLR_GRAY, 2)

    # ── Bottom panel ────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 130), (w, h), CLR_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Letter history
    cv2.putText(frame, "HISTORY", (15, h - 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_GRAY, 1)
    for idx, letter in enumerate(letter_history):
        alpha = 0.4 + (idx / HISTORY_SIZE) * 0.6
        lx    = 80 + idx * 32
        color = tuple(int(c * alpha) for c in CLR_WHITE)
        cv2.putText(frame, letter, (lx, h - 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Word
    word_str = "".join(current_word)
    cv2.putText(frame, "WORD", (15, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_GRAY, 1)
    cv2.putText(frame, word_str if word_str else "...", (70, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLR_WHITE, 2)

    # Sentence
    sentence_str = " ".join(sentence)
    cv2.putText(frame, "TEXT", (15, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_GRAY, 1)
    display_str = sentence_str[-45:] if len(sentence_str) > 45 else sentence_str
    cv2.putText(frame, display_str if display_str else "...", (70, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, CLR_ACCENT, 2)

    # Controls
    cv2.rectangle(frame, (0, h - 22), (w, h), (10, 10, 10), -1)
    cv2.putText(frame,
                "SPACE=confirm word    ENTER=speak    BKSP=delete    C=clear    Q=quit",
                (10, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_GRAY, 1)

    return frame

# ── Main loop ───────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

        prediction = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame, hand_lm,
                mp_hands_module.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            features    = extract_features(hand_lm)
            features_np = np.array(features).reshape(1, -1)
            proba       = model.predict_proba(features_np)[0]
            confidence  = np.max(proba)
            raw_pred    = model.classes_[np.argmax(proba)]

            prediction_buffer.append(raw_pred)

            if len(prediction_buffer) == BUFFER_SIZE:
                most_common = max(set(prediction_buffer),
                                  key=prediction_buffer.count)
                count = prediction_buffer.count(most_common)
                if count >= BUFFER_SIZE * 0.6 and confidence >= CONFIDENCE_MIN:
                    prediction = most_common

            if prediction and prediction == last_added:
                hold_counter = 0
            elif prediction:
                hold_counter += 1
                if hold_counter >= HOLD_FRAMES:
                    current_word.append(prediction)
                    letter_history.append(prediction)
                    last_added   = prediction
                    hold_counter = 0
                    prediction_buffer.clear()

        else:
            hold_counter = 0
            last_added   = None

        # ── Call the HUD ────────────────────────────────
        frame = draw_hud(frame, prediction, confidence, hold_counter,
                         current_word, sentence, letter_history)

        cv2.imshow("ASL Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            word_str = "".join(current_word)
            if current_word:
                sentence.append(word_str)
                current_word = []
                last_added   = None
        elif key == 13:
            full = " ".join(sentence)
            if full:
                speak(full)
                print(f"Speaking: {full}")
        elif key == 8:
            if current_word:
                current_word.pop()
                if letter_history:
                    letter_history.pop()
        elif key == ord('c'):
            current_word = []
            sentence     = []
            last_added   = None
            letter_history.clear()

cap.release()
cv2.destroyAllWindows()