import sys
import cv2
import numpy as np
import joblib
import pyttsx3
import threading
import sounds
from collections import deque
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions import drawing_styles as mp_styles
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout,
                             QProgressBar, QSizePolicy, QShortcut, QSlider,
                             QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence
from utils import extract_features, list_available_camera
from stats_panel import StatsPanel
from autocomplete import AutocompleteEngine

# Config
MODEL_FILE     = "models/az_model.pkl"
BUFFER_SIZE    = 10
CONFIDENCE_MIN = 0.55
HOLD_FRAMES    = 20
HISTORY_SIZE   = 10
SUGGESTION_COUNT = 4

# Colors
C_BG    = "#0f0f0f"
C_PANEL = "#1c1c1c"
C_GREEN = "#00dc64"
C_CYAN  = "#00c8dc"
C_BLUE  = "#00a5ff"
C_WHITE = "#ffffff"
C_GRAY  = "#555555"
C_DARK  = "#2a2a2a"



class CameraThread(QThread):
    frame_ready      = pyqtSignal(np.ndarray)
    prediction_ready = pyqtSignal(str, float, int)

    def __init__(self, model=None, camera_index=0):
        super().__init__()
        self.model = model if model is not None else joblib.load(MODEL_FILE)
        self.running = True
        self.camera_index = camera_index
        self.prediction_buffer = deque(maxlen=BUFFER_SIZE)
        self.hold_counter = 0
        self.last_added = None

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
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

                frame   = cv2.flip(frame, 1)
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                    features    = extract_features(hand_lm)
                    features_np = np.array(features).reshape(1, -1)
                    proba       = self.model.predict_proba(features_np)[0]
                    confidence  = float(np.max(proba))
                    raw_pred    = self.model.classes_[np.argmax(proba)]

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
                        self.last_added   = None
                else:
                    self.hold_counter = 0
                    self.last_added   = None

                self.frame_ready.emit(frame)
                self.prediction_ready.emit(prediction, confidence, self.hold_counter)

        cap.release()

    def confirm_letter(self, letter):
        self.last_added        = letter
        self.hold_counter      = 0
        self.prediction_buffer.clear()

    def stop(self):
        self.running = False
        self.wait()


class SettingsPanel(QWidget):
    settings_applied = pyqtSignal(float, int, int, str, int, int)

    def __init__(self, parent=None, current_camera=0):
        super().__init__(parent)
        self.current_camera = current_camera
        self.setWindowFlags(Qt.Widget)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"background: #141414; border-left: 3px solid {C_GREEN};")
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Header
        header_row = QHBoxLayout()
        title = QLabel("SETTINGS")
        title.setFont(QFont("Courier New", 14, QFont.Bold))
        title.setStyleSheet(f"color: {C_GREEN}; border: none;")
        header_row.addWidget(title)
        header_row.addStretch()
        close_btn = QPushButton("ESC")
        close_btn.setFixedSize(44, 26)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: #2a2a2a; color: {C_GRAY};
                border: none; border-radius: 6px;
                font-family: 'Courier New'; font-size: 11px;
            }}
            QPushButton:hover {{ background: #3a3a3a; color: {C_WHITE}; }}
        """)
        close_btn.clicked.connect(self.hide)
        header_row.addWidget(close_btn)
        layout.addLayout(header_row)

        self._divider(layout)

        # Model file
        layout.addWidget(self._label("MODEL FILE"))
        model_row = QHBoxLayout()
        self.model_input = QLabel("az_model.pkl")
        self.model_input.setFont(QFont("Courier New", 11))
        self.model_input.setStyleSheet(f"""
            background: #1c1c1c; color: {C_WHITE};
            border-radius: 6px; padding: 6px 10px; border: none;
        """)
        self.model_input.setFixedHeight(34)
        browse_btn = QPushButton("...")
        browse_btn.setFixedSize(34, 34)
        browse_btn.setStyleSheet(f"""
            QPushButton {{
                background: #1c1c1c; color: {C_GREEN};
                border: none; border-radius: 6px;
                font-family: 'Courier New'; font-size: 14px;
            }}
            QPushButton:hover {{ background: #2a2a2a; }}
        """)
        browse_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self.model_input, 1)
        model_row.addWidget(browse_btn)
        layout.addLayout(model_row)

        self._divider(layout)

        # Cam selector
        layout.addWidget(self._label("CAMERA"))
        self.camera_combo = QComboBox()
        self.camera_combo.setFixedHeight(34)
        self.camera_combo.setStyleSheet(f"""
            QComboBox {{
                background: #1c1c1c; color: {C_WHITE};
                border-radius: 6px; padding: 6px 10px; border: none;
                font-family: 'Courier New'; font-size: 11px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: #1c1c1c; color: {C_WHITE};
                selection-background-color: {C_GREEN};
                selection-color: #000;
            }}
        """)
        self._populate_cameras()
        layout.addWidget(self.camera_combo)

        refresh_cam_btn = QPushButton("Refresh Cameras")
        refresh_cam_btn.setFixedHeight(30)
        refresh_cam_btn.setStyleSheet(f"""
            QPushButton {{
                background: #1c1c1c; color: {C_GRAY};
                border: none; border-radius: 6px;
                font-family: 'Courier New'; font-size: 10px;
            }}
            QPushButton:hover {{ color: {C_WHITE}; background: #2a2a2a; }}
        """)
        refresh_cam_btn.clicked.connect(self._populate_cameras)
        layout.addWidget(refresh_cam_btn)

        # Confidence slider
        conf_row = QHBoxLayout()
        conf_row.addWidget(self._label("CONFIDENCE THRESHOLD"))
        self.conf_val = QLabel("55%")
        self.conf_val.setFont(QFont("Courier New", 11))
        self.conf_val.setStyleSheet(f"color: {C_GREEN}; border: none;")
        conf_row.addWidget(self.conf_val)
        layout.addLayout(conf_row)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(55)
        self.conf_slider.setStyleSheet(self._slider_style(C_GREEN))
        self.conf_slider.valueChanged.connect(lambda v: self.conf_val.setText(f"{v}%"))
        layout.addWidget(self.conf_slider)
        hint = QHBoxLayout()
        hint.addWidget(self._hint("0% (lenient)"))
        hint.addStretch()
        hint.addWidget(self._hint("100% (strict)"))
        layout.addLayout(hint)

        self._divider(layout)

        # Hold slider
        hold_row = QHBoxLayout()
        hold_row.addWidget(self._label("HOLD SPEED"))
        self.hold_val = QLabel("20 frames")
        self.hold_val.setFont(QFont("Courier New", 11))
        self.hold_val.setStyleSheet(f"color: {C_CYAN}; border: none;")
        hold_row.addWidget(self.hold_val)
        layout.addLayout(hold_row)
        self.hold_slider = QSlider(Qt.Horizontal)
        self.hold_slider.setRange(5, 40)
        self.hold_slider.setValue(20)
        self.hold_slider.setStyleSheet(self._slider_style(C_CYAN))
        self.hold_slider.valueChanged.connect(lambda v: self.hold_val.setText(f"{v} frames"))
        layout.addWidget(self.hold_slider)
        hint2 = QHBoxLayout()
        hint2.addWidget(self._hint("5 (fast)"))
        hint2.addStretch()
        hint2.addWidget(self._hint("40 (slow)"))
        layout.addLayout(hint2)

        self._divider(layout)

        # Buffer slider
        buf_row = QHBoxLayout()
        buf_row.addWidget(self._label("SMOOTHING BUFFER"))
        self.buf_val = QLabel("10 frames")
        self.buf_val.setFont(QFont("Courier New", 11))
        self.buf_val.setStyleSheet(f"color: {C_BLUE}; border: none;")
        buf_row.addWidget(self.buf_val)
        layout.addLayout(buf_row)
        self.buf_slider = QSlider(Qt.Horizontal)
        self.buf_slider.setRange(3, 20)
        self.buf_slider.setValue(10)
        self.buf_slider.setStyleSheet(self._slider_style(C_BLUE))
        self.buf_slider.valueChanged.connect(lambda v: self.buf_val.setText(f"{v} frames"))
        layout.addWidget(self.buf_slider)
        hint3 = QHBoxLayout()
        hint3.addWidget(self._hint("3 (fast)"))
        hint3.addStretch()
        hint3.addWidget(self._hint("20 (smooth)"))
        layout.addLayout(hint3)

        self._divider(layout)

        #Sound section
        sound_row = QHBoxLayout()
        sound_row.addWidget(self._label("SOUND VOLUME"))
        self.sound_val = QLabel("70%")
        self.sound_val.setFont(QFont("Courier New", 11))
        self.sound_val.setStyleSheet(f"color: {C_GREEN}; border:none;")
        sound_row.addWidget(self.sound_val)
        layout.addLayout(sound_row)

        self.sound_slider = QSlider(Qt.Horizontal)
        self.sound_slider.setRange(0, 100)
        self.sound_slider.setValue(70)
        self.sound_slider.setStyleSheet(self._slider_style(C_GREEN))
        self.sound_slider.valueChanged.connect(self._on_volume_change)
        layout.addWidget(self.sound_slider)

        self.mute_btn = QPushButton("Sound ON")
        self.mute_btn.setFixedHeight(36)
        self.mute_btn.setCheckable(True)
        self.mute_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_DARK}; color: {C_WHITE};
                border: none; border-radius: 6px;
                font-family: 'Courier New'; font-size: 12px;
            }}
            QPushButton:checked {{ background: #ff5f57; color: #000; }}
        """)
        self.mute_btn.clicked.connect(self._toggle_mute)
        layout.addWidget(self.mute_btn)

        self._divider(layout)

        layout.addWidget(self._label("STARTUP BEHAVIOR"))

        self.autostart_camera_btn = QPushButton("Auto-start Camera: ON")
        self.autostart_camera_btn.setFixedHeight(36)
        self.autostart_camera_btn.setCheckable(True)
        self.autostart_camera_btn.setChecked(True)
        self.autostart_camera_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_GREEN}; color: #000;
                border: none; border-radius: 6px;
                font-family: 'Courier New'; font-size: 12px;
            }}
            QPushButton:!checked {{ background: {C_DARK}; color: {C_WHITE}; }}
        """)
        self.autostart_camera_btn.clicked.connect(self._toggle_autostart_label)
        layout.addWidget(self.autostart_camera_btn)

        self.fullscreen_btn = QPushButton("Start Fullscreen: ON")
        self.fullscreen_btn.setFixedHeight(36)
        self.fullscreen_btn.setCheckable(True)
        self.fullscreen_btn.setChecked(True)
        self.fullscreen_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_GREEN}; color: #000;
                border: none; border-radius: 6px;
                font-family: 'Courier New'; font-size: 12px;
            }}
            QPushButton:!checked {{ background: {C_DARK}; color: {C_WHITE}; }}
        """)
        self.fullscreen_btn.clicked.connect(self._toggle_fullscreen_label)
        layout.addWidget(self.fullscreen_btn)

        self._divider(layout)

        sugg_row = QHBoxLayout()
        sugg_row.addWidget(self._label("SUGGESTION COUNT"))
        self.sugg_val = QLabel("4")
        self.sugg_val.setFont(QFont("Courier New", 11))
        self.sugg_val.setStyleSheet(f"color: {C_BLUE}; border: none;")
        sugg_row.addWidget(self.sugg_val)
        layout.addLayout(sugg_row)

        self.sugg_slider = QSlider(Qt.Horizontal)
        self.sugg_slider.setRange(1, 6)
        self.sugg_slider.setValue(4)
        self.sugg_slider.setStyleSheet(self._slider_style(C_BLUE))
        self.sugg_slider.valueChanged.connect(lambda v: self.sugg_val.setText(str(v)))
        layout.addWidget(self.sugg_slider)
    
        # Apply button
        apply_btn = QPushButton("APPLY SETTINGS")
        apply_btn.setFixedHeight(44)
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_GREEN}; color: #000;
                border: none; border-radius: 8px;
                font-family: 'Courier New'; font-size: 13px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #00b850; }}
            QPushButton:pressed {{ background: #009040; }}
        """)
        apply_btn.clicked.connect(self._apply)
        layout.addWidget(apply_btn)

        # Reset + Close
        btn_row = QHBoxLayout()
        for text, slot in [("RESET", self._reset), ("CLOSE", self.hide)]:
            btn = QPushButton(text)
            btn.setFixedHeight(34)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: #1c1c1c; color: {C_GRAY};
                    border: none; border-radius: 6px;
                    font-family: 'Courier New'; font-size: 11px;
                }}
                QPushButton:hover {{ color: {C_WHITE}; background: #2a2a2a; }}
            """)
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)
        layout.addStretch()

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Courier New", 10))
        lbl.setStyleSheet(f"color: {C_GRAY}; border: none;")
        return lbl

    def _hint(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Courier New", 9))
        lbl.setStyleSheet("color: #444; border: none;")
        return lbl

    def _divider(self, layout):
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background: #2a2a2a; border: none;")
        layout.addWidget(line)

    def _slider_style(self, color):
        return f"""
            QSlider::groove:horizontal {{
                height: 6px; background: #2a2a2a; border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: {color}; border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                width: 16px; height: 16px; margin: -5px 0;
                background: {color}; border-radius: 8px;
            }}
        """

    def _browse_model(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "models/", "Model files (*.pkl *.h5)"
        )
        if path:
            self.model_input.setText(path.split("/")[-1])
            self._model_path = path

    def _apply(self):
        confidence = self.conf_slider.value() / 100.0
        hold = self.hold_slider.value()
        buffer = self.buf_slider.value()
        camera_index = self.get_selected_camera()
        suggestion_count = self.get_suggestion_count()
        self._model_path = getattr(self, '_model_path',
                                   f"models/{self.model_input.text()}")
        self.settings_applied.emit(confidence, hold, buffer, self._model_path, camera_index, suggestion_count)
        self.hide()

    def _reset(self):
        self.conf_slider.setValue(55)
        self.hold_slider.setValue(20)
        self.buf_slider.setValue(10)
        self.sound_slider.setValue(70)
        self.sugg_slider.setValue(4)
        self.autostart_camera_btn.setChecked(True)
        self._toggle_autostart_label(True)
        self.fullscreen_btn.setChecked(True)
        self._toggle_fullscreen_label(True)
        self.model_input.setText("az_model.pkl")
        if hasattr(self, '_model_path'):
            del self._model_path

    def _on_volume_change(self, value):
        import sounds
        self.sound_val.setText(f"{value}%")
        sounds.set_volume(value / 100.0)

    def _toggle_mute(self, checked):
        import sounds
        sounds.set_muted(checked)
        self.mute_btn.setText("Sound OFF" if checked else "Sound ON")

    def _populate_cameras(self):
        from utils import list_available_camera
        self.camera_combo.clear()
        cameras = list_available_camera(skip_index=self.current_camera)
        for idx in cameras:
            label = f"Camera {idx}" + (" (active)" if idx == self.current_camera else "")
            self.camera_combo.addItem(label, idx)
            if idx == self.current_camera:
                self.camera_combo.setCurrentIndex(self.camera_combo.count() - 1)

    def get_selected_camera(self):
        return self.camera_combo.currentData()

    def _toggle_autostart_label(self, checked):
        self.autostart_camera_btn.setText(f"Auto-start Camera: {'ON' if checked else 'OFF'}")

    def _toggle_fullscreen_label(self, checked):
        self.fullscreen_btn.setText(f"Start Fullscreen: {'ON' if checked else 'OFF'}")

    def get_autostart_camera(self):
        return self.autostart_camera_btn.isChecked()

    def get_start_fullscreen(self):
        return self.fullscreen_btn.isChecked()

    def get_suggestion_count(self):
        return self.sugg_slider.value()
    
class MainWindow(QMainWindow):
    def __init__(self, model=None):
        super().__init__()
        self.model          = model if model is not None else joblib.load(MODEL_FILE)
        self.engine         = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.current_word   = []
        self.sentence       = []
        self.letter_history = deque(maxlen=HISTORY_SIZE)
        self.last_prediction = ""
        self.last_confidence = 0.0
        self.last_hold       = 0

        self.setWindowTitle("AzSL Recognition")
        self.showFullScreen()
        self.setStyleSheet(f"background-color: {C_BG}; color: {C_WHITE};")

        self._build_ui()
        self._setup_shortcuts()

        self.settings_panel = SettingsPanel(self, current_camera=0)
        self.settings_panel.settings_applied.connect(self._apply_settings)
        self.settings_panel.hide()

        self.camera_thread = CameraThread(self.model)
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.prediction_ready.connect(self.update_prediction)
        self.camera_thread.start()

        # Stats panel
        self.stats_panel = StatsPanel(self)
        self.stats_panel.hide()
        self._reposition_stats()

        self.autocomplete = AutocompleteEngine()


    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # Title bar
        title_bar = QHBoxLayout()
        title = QLabel("AzSL Recognition")
        title.setFont(QFont("Courier New", 14))
        title.setStyleSheet(f"color: {C_GRAY};")
        title_bar.addWidget(title)
        title_bar.addStretch()

        # stats button
        stats_btn = QPushButton("Stats")
        stats_btn.setFixedSize(100, 32)
        stats_btn.setStyleSheet(f"""
            QPushButton {{
                background: #2a2a2a; color: {C_GRAY};
                border: none; border-radius: 6px;
                font-family: 'Courier New'; font-size: 12px;
            }}
            QPushButton:hover {{ color: {C_WHITE}; background: #3a3a3a; }}
        """)
        stats_btn.clicked.connect(self._toggle_stats)
        title_bar.addWidget(stats_btn)
        
        # Settings button — added BEFORE root.addLayout
        settings_btn = QPushButton("⚙  Settings")
        settings_btn.setFixedSize(110, 32)
        settings_btn.setStyleSheet(f"""
            QPushButton {{
                background: #2a2a2a; color: {C_GRAY};
                border: none; border-radius: 6px;
                font-family: 'Courier New'; font-size: 12px;
            }}
            QPushButton:hover {{ color: {C_WHITE}; background: #3a3a3a; }}
        """)
        settings_btn.clicked.connect(self._toggle_settings)
        title_bar.addWidget(settings_btn)

        quit_btn = QPushButton("✕  Quit")
        quit_btn.setFixedSize(90, 32)
        quit_btn.setStyleSheet(f"""
            QPushButton {{
                background: #ff5f57; color: #000;
                border: none; border-radius: 6px;
                font-family: 'Courier New'; font-size: 12px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #ff3b30; }}
        """)
        quit_btn.clicked.connect(self.close)
        title_bar.addWidget(quit_btn)
        root.addLayout(title_bar)

        # Main row
        main_row = QHBoxLayout()
        main_row.setSpacing(12)

        self.cam_label = QLabel()
        self.cam_label.setMinimumSize(700, 480)
        self.cam_label.setStyleSheet(f"""
            background: {C_PANEL};
            border-radius: 12px;
            border: 1px solid #2a2a2a;
        """)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_row.addWidget(self.cam_label, 65)

        right = QVBoxLayout()
        right.setSpacing(10)

        # Letter box
        letter_panel = QWidget()
        letter_panel.setMinimumHeight(220)
        letter_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 12px; border: 1px solid #2a2a2a;")
        letter_layout = QVBoxLayout(letter_panel)
        letter_layout.setContentsMargins(16, 12, 16, 12)
        lbl_title = QLabel("DETECTED LETTER")
        lbl_title.setFont(QFont("Courier New", 10))
        lbl_title.setStyleSheet(f"color: {C_GRAY}; border: none;")
        self.letter_label = QLabel("—")
        self.letter_label.setFont(QFont("Courier New", 100, QFont.Bold))
        self.letter_label.setStyleSheet(f"color: {C_GREEN}; border: none;")
        self.letter_label.setAlignment(Qt.AlignCenter)
        self.letter_label.setMinimumHeight(140)
        letter_layout.addWidget(lbl_title)
        letter_layout.addWidget(self.letter_label)
        right.addWidget(letter_panel)

        # Confidence bar
        conf_panel = QWidget()
        conf_panel.setMinimumHeight(80)
        conf_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 12px; border: 1px solid #2a2a2a;")
        conf_layout = QVBoxLayout(conf_panel)
        conf_layout.setContentsMargins(16, 10, 16, 10)
        conf_lbl = QLabel("CONFIDENCE")
        conf_lbl.setFont(QFont("Courier New", 10))
        conf_lbl.setStyleSheet(f"color: {C_GRAY}; border: none;")
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setTextVisible(True)
        self.conf_bar.setFixedHeight(28)
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C_DARK}; border-radius: 6px; border: none;
                color: #000; font-family: 'Courier New'; font-size: 12px; font-weight: bold;
            }}
            QProgressBar::chunk {{ background: {C_GREEN}; border-radius: 6px; }}
        """)
        conf_layout.addWidget(conf_lbl)
        conf_layout.addWidget(self.conf_bar)
        right.addWidget(conf_panel)

        # Hold bar
        hold_panel = QWidget()
        hold_panel.setMinimumHeight(80)
        hold_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 12px; border: 1px solid #2a2a2a;")
        hold_layout = QVBoxLayout(hold_panel)
        hold_layout.setContentsMargins(16, 10, 16, 10)
        hold_lbl = QLabel("HOLD PROGRESS")
        hold_lbl.setFont(QFont("Courier New", 10))
        hold_lbl.setStyleSheet(f"color: {C_GRAY}; border: none;")
        self.hold_bar = QProgressBar()
        self.hold_bar.setRange(0, HOLD_FRAMES)
        self.hold_bar.setTextVisible(False)
        self.hold_bar.setFixedHeight(28)
        self.hold_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C_DARK}; border-radius: 6px; border: none;
            }}
            QProgressBar::chunk {{ background: {C_CYAN}; border-radius: 6px; }}
        """)
        hold_layout.addWidget(hold_lbl)
        hold_layout.addWidget(self.hold_bar)
        right.addWidget(hold_panel)

        # History
        hist_panel = QWidget()
        hist_panel.setMinimumHeight(80)
        hist_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 12px; border: 1px solid #2a2a2a;")
        hist_layout = QVBoxLayout(hist_panel)
        hist_layout.setContentsMargins(16, 10, 16, 10)
        hist_lbl = QLabel("HISTORY")
        hist_lbl.setFont(QFont("Courier New", 10))
        hist_lbl.setStyleSheet(f"color: {C_GRAY}; border: none;")
        self.history_label = QLabel("—")
        self.history_label.setFont(QFont("Courier New", 18))
        self.history_label.setStyleSheet(f"color: {C_WHITE}; border: none;")
        hist_layout.addWidget(hist_lbl)
        hist_layout.addWidget(self.history_label)
        right.addWidget(hist_panel)

        right.addStretch()
        main_row.addLayout(right, 35)
        root.addLayout(main_row, 1)

        # Word row
        word_panel = QWidget()
        word_panel.setFixedHeight(64)
        word_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 10px; border: 1px solid #2a2a2a;")
        word_row = QHBoxLayout(word_panel)
        word_row.setContentsMargins(20, 0, 20, 0)
        word_lbl = QLabel("WORD")
        word_lbl.setFixedWidth(90)
        word_lbl.setFont(QFont("Courier New", 10))
        word_lbl.setStyleSheet(f"color: {C_GRAY}; border: none;")
        self.word_label = QLabel("...")
        self.word_label.setFont(QFont("Courier New", 24))
        self.word_label.setStyleSheet(f"color: {C_WHITE}; border: none;")
        word_row.addWidget(word_lbl)
        word_row.addWidget(self.word_label)
        word_row.addStretch()
        root.addWidget(word_panel)

        # Suggestions row
        self.suggestions_panel = QWidget()
        self.suggestions_panel.setFixedHeight(44)
        self.suggestions_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 10px; border: 1px solid #2a2a2a;")
        sugg_layout = QHBoxLayout(self.suggestions_panel)
        sugg_layout.setContentsMargins(16, 0, 16, 0)
        sugg_layout.setSpacing(10)

        sugg_label = QLabel("SUGGEST")
        sugg_label.setFixedWidth(90)
        sugg_label.setFont(QFont("Courier New", 10))
        sugg_label.setStyleSheet(f"color: {C_GRAY}; border: none;")
        sugg_layout.addWidget(sugg_label)

        self.suggestion_buttons = []
        for i in range(6):
            btn = QPushButton("")
            btn.setFixedHeight(30)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {C_DARK}; color: {C_WHITE};
                    border: none; border-radius: 6px;
                    font-family: 'Courier New'; font-size: 12px;
                    padding: 0 12px;
                }}
                QPushButton:hover {{ background: {C_GREEN}; color: #000; }}
            """)
            btn.hide()
            btn.clicked.connect(lambda checked, idx=i: self._accept_suggestion(idx))
            sugg_layout.addWidget(btn)
            self.suggestion_buttons.append(btn)

        sugg_layout.addStretch()
        root.addWidget(self.suggestions_panel)


        # Sentence row
        sent_row = QHBoxLayout()
        sent_row.setSpacing(10)
        sent_panel = QWidget()
        sent_panel.setFixedHeight(64)
        sent_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 10px; border: 1px solid #2a2a2a;")
        sent_layout = QHBoxLayout(sent_panel)
        sent_layout.setContentsMargins(20, 0, 20, 0)
        sent_lbl = QLabel("SENTENCE")
        sent_lbl.setFixedWidth(90)
        sent_lbl.setFont(QFont("Courier New", 10))
        sent_lbl.setStyleSheet(f"color: {C_GRAY}; border: none;")
        self.sent_label = QLabel("...")
        self.sent_label.setFont(QFont("Courier New", 18))
        self.sent_label.setStyleSheet(f"color: {C_BLUE}; border: none;")
        sent_layout.addWidget(sent_lbl)
        sent_layout.addWidget(self.sent_label)
        sent_layout.addStretch()
        sent_row.addWidget(sent_panel, 1)

        speak_btn = QPushButton("SPEAK")
        speak_btn.setFixedSize(110, 64)
        speak_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_GREEN}; color: #000;
                border: none; border-radius: 10px;
                font-family: 'Courier New'; font-size: 15px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #00b850; }}
            QPushButton:pressed {{ background: #009040; }}
        """)
        speak_btn.clicked.connect(self.speak_sentence)
        sent_row.addWidget(speak_btn)
        root.addLayout(sent_row)

        export_btn = QPushButton("EXPORT")
        export_btn.setFixedSize(110, 64)
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_DARK}; color: {C_WHITE};
                border: none; border-radius: 10px;
                font-family: 'Courier New'; font-size: 14px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #3a3a3a; }}
            QPushButton:pressed {{ background: #2a2a2a; }}
        """)
        export_btn.clicked.connect(self.export_session)
        sent_row.addWidget(export_btn)

        # Controls bar
        controls = QLabel(
            "SPACE = confirm word     TAB = accept suggestion     BACKSPACE = delete letter     C = clear all     Ctrl+E = export     ESC = quit"
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
        QShortcut(QKeySequence("Ctrl+E"), self, self.export_session)
        QShortcut(QKeySequence("Tab"), self, lambda: self._accept_suggestion(0))

    def _toggle_settings(self):
        if self.settings_panel.isVisible():
            self.settings_panel.hide()
        else:
            self._reposition_settings()
            self.settings_panel.raise_()
            self.settings_panel.show()
    
    def _toggle_stats(self):
        if self.stats_panel.isVisible():
            self.stats_panel.hide()
        else:
            self._reposition_stats()
            self.stats_panel.show()
            self.stats_panel.raise_()

    def _reposition_stats(self):
        panel_w = 300
        self.stats_panel.setGeometry(self.width() - panel_w, 0, panel_w, self.height())

    def _reposition_settings(self):
        panel_w = 320
        panel_h = self.height()
        screen_x = self.width() - panel_w
        self.settings_panel.setGeometry(screen_x, 0, panel_w, panel_h)
        self.settings_panel.setFixedSize(panel_w, self.height())
        self.settings_panel.move(self.width() - panel_w, 0)

    def _apply_settings(self, confidence, hold, buffer, model_path, camera_index, suggestion_count):
        global CONFIDENCE_MIN, HOLD_FRAMES, BUFFER_SIZE, SUGGESTION_COUNT
        CONFIDENCE_MIN = confidence
        HOLD_FRAMES = hold
        BUFFER_SIZE = buffer
        SUGGESTION_COUNT= suggestion_count
        self.hold_bar.setRange(0, hold)

        if camera_index != self.camera_thread.camera_index:
            self.camera_thread.stop()
            self.camera_thread = CameraThread(self.model, camera_index=camera_index)
            self.camera_thread.frame_ready.connect(self.update_frame)
            self.camera_thread.prediction_ready.connect(self.update_prediction)
            self.camera_thread.start()
            self.settings_panel.current_camera = camera_index
        else:
            self.camera_thread.prediction_buffer = deque(maxlen=buffer)

        self._update_suggestions()
        print(f"Settings applied: conf={confidence} hold={hold} buffer={buffer} camera={camera_index}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'settings_panel') and self.settings_panel.isVisible():
            self._reposition_settings()
        if hasattr(self, 'stats_panel') and self.stats_panel.isVisible():
            self._reposition_stats()

    def update_frame(self, frame):
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img   = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix   = QPixmap.fromImage(img).scaled(
            self.cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.cam_label.setPixmap(pix)

    def update_prediction(self, prediction, confidence, hold_counter):
        self.last_prediction = prediction
        self.last_confidence = confidence
        self.last_hold       = hold_counter

        if prediction:
            self.letter_label.setText(prediction)
            conf_pct  = int(confidence * 100)
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

            if hold_counter >= HOLD_FRAMES:
                self.current_word.append(prediction)
                self.letter_history.append(prediction)
                self.camera_thread.confirm_letter(prediction)
                sounds.play_letter_confirm()
                self.stats_panel.record_letter(prediction)
                self._refresh_display()
        else:
            self.letter_label.setText("—")
            self.conf_bar.setValue(0)
            self.hold_bar.setValue(0)

    def _refresh_display(self):
        self.word_label.setText("".join(self.current_word) or "...")
        sentence_str = " ".join(self.sentence)
        self.sent_label.setText(
            sentence_str[-50:] if len(sentence_str) > 50 else sentence_str or "..."
        )
        self.history_label.setText(
            "  ".join(list(self.letter_history)) or "—"
        )
        self._update_suggestions()

    def _update_suggestions(self):
        prefix = "".join(self.current_word)
        suggestions = self.autocomplete.suggest(prefix, max_results=SUGGESTION_COUNT) if prefix else []

        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(suggestions):
                btn.setText(suggestions[i])
                btn.show()
            else:
                btn.hide()

    def _accept_suggestion(self, idx):
        prefix = "".join(self.current_word)
        suggestions = self.autocomplete.suggest(prefix, max_results=4)
        if idx < len(suggestions):
            chosen = suggestions[idx]
            self.sentence.append(chosen)
            self.current_word = []
            self.letter_history.clear()
            sounds.play_word_complete()
            self.stats_panel.record_word()
            self._refresh_display()

    def confirm_word(self):
        if self.current_word:
            self.sentence.append("".join(self.current_word))
            self.current_word = []
            sounds.play_word_complete()
            self.stats_panel.record_word()
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

    def export_session(self):
        from datetime import datetime
        import os

        os.makedirs("exports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = f"exports/session_{timestamp}.txt"

        full_sentence = " ".join(self.sentence)
        if self.current_word:
            full_sentence += (" " if full_sentence else "") + "".join(self.current_word)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("AzSl Recognition - Session Export\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 40 + "\n\n")
            f.write("TRANSCRIPT:\n")
            f.write(full_sentence if full_sentence else "(empty)")
            f.write("\n\n")
            f.write("=" * 40 + "\n")
            f.write(f"Words completed: {len(self.sentence)}\n")
            f.write(f"Letters signed: {len(self.letter_history)}\n")

        print(f"Session exported to: {filepath}")
        self._show_export_toast(filepath)

    def _show_export_toast(self, filepath):
        toast = QLabel(f"Saved to {filepath}", self)
        toast.setFont(QFont("Courier New", 11))
        toast.setStyleSheet(f"""
            background: {C_GREEN}; color: #000;
            padding: 10px 16px; border-radius: 8px;
        """)
        toast.adjustSize()
        toast.move((self.width() - toast.width()) // 2, self.height() - 140)
        toast.show()
        QTimer.singleShot(2500, toast.deleteLater)
         
    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    sys.exit(app.exec_())