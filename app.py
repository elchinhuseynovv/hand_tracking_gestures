import sys
import cv2
import numpy as np
import joblib
import pyttsx3
import threading
from collections import deque
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions import drawing_styles as mp_styles
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout,
                             QProgressBar, QSizePolicy, QShortcut)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence, QColor
from utils import extract_features

# ── Config ─────────────────────────────────────────────
MODEL_FILE     = "models/az_model.pkl"
BUFFER_SIZE    = 10
CONFIDENCE_MIN = 0.55
HOLD_FRAMES    = 20
HISTORY_SIZE   = 10
# ───────────────────────────────────────────────────────

# ── Colors ─────────────────────────────────────────────
C_BG      = "#0f0f0f"
C_PANEL   = "#1c1c1c"
C_GREEN   = "#00dc64"
C_CYAN    = "#00c8dc"
C_BLUE    = "#00a5ff"
C_WHITE   = "#ffffff"
C_GRAY    = "#555555"
C_DARK    = "#2a2a2a"
# ───────────────────────────────────────────────────────

class CameraThread(QThread):
    frame_ready      = pyqtSignal(np.ndarray)
    prediction_ready = pyqtSignal(str, float, int)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.running = True
        self.prediction_buffer = deque(maxlen=BUFFER_SIZE)
        self.hold_counter = 0
        self.last_added = None
        self.confirmed_letter = pyqtSignal(str)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        with mp_hands_module.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6
        ) as hands:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                prediction = ""
                confidence = 0.0

                if results.multi_hand_landmarks:
                    hand_lm = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(
                        frame, hand_lm,
                        mp_hands_module.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    features = extract_features(hand_lm)
                    features_np = np.array(features).reshape(1, -1)
                    proba = self.model.predict_proba(features_np)[0]
                    confidence = float(np.max(proba))
                    raw_pred = self.model.classes_[np.argmax(proba)]

                    self.prediction_buffer.append(raw_pred)

                    if len(self.prediction_buffer) == BUFFER_SIZE:
                        most_common = max(set(self.prediction_buffer),
                                          key=self.prediction_buffer.count)
                        count = self.prediction_buffer.count(most_common)
                        if count >= BUFFER_SIZE * 0.6 and confidence >= CONFIDENCE_MIN:
                            prediction = most_common

                    if prediction and prediction == self.last_added:
                        self.hold_counter = 0
                    elif prediction:
                        self.hold_counter += 1
                    else:
                        self.hold_counter = 0
                        self.last_added = None

                else:
                    self.hold_counter = 0
                    self.last_added = None

                self.frame_ready.emit(frame)
                self.prediction_ready.emit(prediction, confidence,
                                           self.hold_counter)

            cap.release()
            
    def confirm_letter(self, letter):
        self.last_added = letter
        self.hold_counter = 0
        self.prediction_buffer.clear()
    
    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = joblib.load(MODEL_FILE)
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.current_word = []
        self.sentence = []
        self.letter_history = deque(maxlen=HISTORY_SIZE)
        self.last_prediction = ""
        self.last_confidence = 0.0
        self.last_hold = 0

        self.setWindowTitle("AzSL Recognition")
        self.showFullScreen()
        self.setStyleSheet(f"background-color: {C_BG}; color: {C_WHITE};")

        self._build_ui()
        self._setup_shortcuts()

        self.camera_thread = CameraThread(self.model)
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.prediction_ready.connect(self.update_prediction)
        self.camera_thread.start()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # TITLE BAR
        title_bar = QHBoxLayout()
        title = QLabel("AzSL Recognition")
        title.setFont(QFont("Courier New", 13))
        title.setStyleSheet(f"color: {C_GRAY};")
        title_bar.addWidget(title)
        title_bar.addStretch()
        quit_btn = QPushButton("X Quit")
        quit_btn.setFixedSize(80, 28)
        quit_btn.setStyleSheet(f"""
            QPushButton {{
                background: #ff5f57; color: {C_BG};
                border: none; border-radius: 6px;
                font-family: 'Courier New'; font-size: 12px;
            }}
            QPushButton:hover {{ background: #ff3b30; }}
        """)
        quit_btn.clicked.connect(self.close)
        title_bar.addWidget(quit_btn)
        root.addLayout(title_bar)

        # MAIN ROW
        main_row = QHBoxLayout()
        main_row.setSpacing(8)

        # WEBCAM FEED
        self.cam_label = QLabel()
        self.cam_label.setMinimumSize(640, 400)
        self.cam_label.setStyleSheet(f"background: {C_PANEL}; border-radius: 8px;")
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_row.addWidget(self.cam_label, 3)

        # RIGHT PANEL
        right = QVBoxLayout()
        right.setSpacing(8)

        # LETTER DISPLAY
        letter_panel = QWidget()
        letter_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 8px;")
        letter_layout = QVBoxLayout(letter_panel)
        lbl_title = QLabel("DETECETED LETTER")
        lbl_title.setFont(QFont("Courier New", 10))
        lbl_title.setStyleSheet(f"color: {C_GREEN};")
        self.letter_label = QLabel("-")
        self.letter_label.setFont(QFont("Courier New", 80, QFont.Bold))
        self.letter_label.setStyleSheet(f"color: {C_GREEN};")
        self.letter_label.setAlignment(Qt.AlignCenter)
        letter_layout.addWidget(lbl_title)
        letter_layout.addWidget(self.letter_label)
        right.addWidget(letter_panel, 2)

        # CONFIDENCE BAR
        conf_panel = QWidget()
        conf_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 8px")
        conf_layout = QVBoxLayout(conf_panel)
        conf_lbl = QLabel("CONFIDENCE")
        conf_lbl.setFont(QFont("Courier New", 10))
        conf_lbl.setStyleSheet(f"color: {C_GRAY};")
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setTextVisible(True)
        self.conf_bar.setFixedHeight(20)
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C_DARK}; border-radius: 4px;
                color: {C_BG}; font-family: 'Courier New'; font-size: 11px;
            }}
            QProgressBar::chunk {{ background: {C_GREEN}; border-radius: 4px; }}
        """)
        conf_layout.addWidget(conf_lbl)
        conf_layout.addWidget(self.conf_bar)
        right.addWidget(conf_panel)

        # HOLD BAR
        hold_panel = QWidget()
        hold_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 8px;")
        hold_layout = QVBoxLayout(hold_panel)
        hold_lbl = QLabel("HOLD PROGRESS")
        hold_lbl.setFont(QFont("Courier New", 10))
        hold_lbl.setStyleSheet(f"color: {C_GRAY};")
        self.hold_bar = QProgressBar()
        self.hold_bar.setRange(0, HOLD_FRAMES)
        self.hold_bar.setTextVisible(False)
        self.hold_bar.setFixedHeight(20)
        self.hold_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C_DARK}; border-radius: 4px;
            }}
            QProgressBar::chunk {{ background: {C_CYAN}; border-radius: 4px; }}
        """)
        hold_layout.addWidget(hold_lbl)
        hold_layout.addWidget(self.hold_bar)
        right.addWidget(hold_panel)

        # HISTORY
        hist_panel = QWidget()
        hist_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 8px;")
        hist_layout = QVBoxLayout(hist_panel)
        hist_lbl = QLabel("HISTORY")
        hist_lbl.setFont(QFont("Courier New", 10))
        hist_lbl.setStyleSheet(f"color: {C_GRAY};")
        self.history_label = QLabel("—")
        self.history_label.setFont(QFont("Courier New", 16))
        self.history_label.setStyleSheet(f"color: {C_WHITE}; letter-spacing: 6px;")
        hist_layout.addWidget(hist_lbl)
        hist_layout.addWidget(self.history_label)
        right.addWidget(hist_panel)

        right.addStretch()
        main_row.addLayout(right, 1)
        root.addLayout(main_row, 1)

        # ── Word row ───────────────────────────────────
        word_panel = QWidget()
        word_panel.setFixedHeight(60)
        word_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 8px;")
        word_row = QHBoxLayout(word_panel)
        word_lbl = QLabel("WORD")
        word_lbl.setFont(QFont("Courier New", 10))
        word_lbl.setStyleSheet(f"color: {C_GRAY};")
        self.word_label = QLabel("...")
        self.word_label.setFont(QFont("Courier New", 22))
        self.word_label.setStyleSheet(f"color: {C_WHITE};")
        word_row.addWidget(word_lbl)
        word_row.addWidget(self.word_label)
        word_row.addStretch()
        root.addWidget(word_panel)

        # ── Sentence row ───────────────────────────────
        sent_row = QHBoxLayout()
        sent_panel = QWidget()
        sent_panel.setFixedHeight(52)
        sent_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 8px;")
        sent_layout = QHBoxLayout(sent_panel)
        sent_lbl = QLabel("SENTENCE")
        sent_lbl.setFont(QFont("Courier New", 10))
        sent_lbl.setStyleSheet(f"color: {C_GRAY};")
        self.sent_label = QLabel("...")
        self.sent_label.setFont(QFont("Courier New", 16))
        self.sent_label.setStyleSheet(f"color: {C_BLUE};")
        sent_layout.addWidget(sent_lbl)
        sent_layout.addWidget(self.sent_label)
        sent_layout.addStretch()
        sent_row.addWidget(sent_panel, 1)

        speak_btn = QPushButton("SPEAK")
        speak_btn.setFixedSize(100, 52)
        speak_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_GREEN}; color: {C_BG};
                border: none; border-radius: 8px;
                font-family: 'Courier New'; font-size: 14px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #00b850; }}
            QPushButton:pressed {{ background: #009040; }}
        """)
        speak_btn.clicked.connect(self.speak_sentence)
        sent_row.addWidget(speak_btn)
        root.addLayout(sent_row)

        # ── Controls bar ───────────────────────────────
        controls = QLabel(
            "SPACE = confirm word     BACKSPACE = delete letter     C = clear all     ESC = quit"
        )
        controls.setFont(QFont("Courier New", 10))
        controls.setStyleSheet(f"color: {C_GRAY};")
        controls.setAlignment(Qt.AlignCenter)
        root.addWidget(controls)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Escape"), self, self.close)
        QShortcut(QKeySequence("Space"), self, self.confirm_word)
        QShortcut(QKeySequence("Backspace"), self, self.delete_letter)
        QShortcut(QKeySequence("Return"), self, self.speak_sentence)
        QShortcut(QKeySequence("c"), self, self.clear_all)

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.cam_label.setPixmap(pix)

    def update_prediction(self, prediction, confidence, hold_counter):
        self.last_prediction = prediction
        self.last_confidence = confidence
        self.last_hold = hold_counter

        if prediction:
            self.letter_label.setText(prediction)
            conf_pct = int(confidence * 100)
            self.conf_bar.setValue(conf_pct)
            self.conf_bar.setFormat(f"{conf_pct}%")
            color = C_GREEN if confidence > 0.75 else "#00a5ff"
            self.conf_bar.setStyleSheet(f"""
                QProgressBar {{
                    background: {C_DARK}; border-radius: 4px;
                    color: {C_BG}; font-family: 'Courier New'; font-size: 11px;
                }}
                QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}
            """)
            self.hold_bar.setValue(hold_counter)

            # auto confirm letter when hold completes
            if  hold_counter >= HOLD_FRAMES:
                self.current_word.append(prediction)
                self.letter_history.append(prediction)
                self.camera_thread.confirm_letter(prediction)
                self._refresh_display()
        
        else:
            self.letter_label.setText("-")
            self.conf_bar.setValue(0)
            self.hold_bar.setValue(0)

    def _refresh_display(self):
        self.word_label.setText("".join(self.current_word) or "...")
        sentence_str = " ".join(self.sentence)
        self.sent_label.setText(sentence_str[-50:] if len(sentence_str) > 50
                                else sentence_str or "...")
        self.history_label.setText(
            " ".join(list(self.letter_history)) or "-"
        )

    def confirm_word(self):
        if self.current_word:
            self.sentence.append("".join(self.current_word))
            self.current_word = []
            self._refresh_display()

    def delete_letter(self):
        if self.current_word:
            self.current_word.pop()
            if self.letter_history:
                self.letter_history.pop()
            self._refresh_display()

    def clear_all(self):
        self.current_word = []
        self.sentence     = []
        self.letter_history.clear()
        self._refresh_display()


    def speak_sentence(self):
        full = " ".join(self.sentence)
        if full:
            def _speak():
                self.engine.say(full)
                self.engine.runAndWait()
            threading.Thread(target=_speak, daemon=True).start()
            print(f"Speaking: {full}")
    
    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    sys.exit(app.exec_())