import cv2
import mediapipe as mp

from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_drawing

mp_hands = mp_hands_module
mp_draw = mp_drawing

def fingers_up(landmarks):
    tips = [8, 12, 16, 20]  # index, mid, ring, pinky tips
    fingers = []

    # thumb - compare x-axis
    fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)

    # other fingers - compare y-axis: tip above pip joint = up
    for tip in tips:
        fingers.append(1 if landmarks[tip].y < landmarks[tip - 2].y else 0)

    return fingers  # thumb, index, mid, ring, pinky

def detect_gesture(fingers):
    gestures = {
        (0,0,0,0,0): "Fist ✊",
        (1,1,1,1,1): "Open Hand 🖐️",
        (0,1,0,0,0): "Pointing ☝️",
        (0,1,1,0,0): "Peace ✌️",
        (1,1,0,0,1): "Hang Loose 🤙",
        (1,0,0,0,1): "Rock On 🤘",
        (0,1,1,1,1): "Four Fingers",
        (1,1,1,1,0): "Four (Thumb)",
        (0,0,0,0,1): "Pinky",
        (1,0,0,0,0): "Thumb",
    }
    return gestures.get(tuple(fingers), "Unknown Gesture")

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                lm = hand_lm.landmark
                fingers = fingers_up(lm)
                gesture = detect_gesture(fingers)

                h, w, _ = frame.shape
                x = int(lm[0].x * w)
                y = int(lm[0].y * h) - 20
                cv2.putText(frame, gesture, (x - 60, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)

        cv2.imshow("Hand Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()