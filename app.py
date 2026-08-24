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
                             QComboBox, QTabWidget, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence
from utils import extract_features, list_available_camera
from stats_panel import StatsPanel
from autocomplete import AutocompleteEngine
from translations import t, set_language, get_language
from settings_store import load_settings, save_settings

# Config
MODEL_FILE     = "models/az_model.pkl"
BUFFER_SIZE    = 10
CONFIDENCE_MIN = 0.55
HOLD_FRAMES    = 20
HISTORY_SIZE   = 10
SUGGESTION_COUNT = 4

# Colors
C_BG      = "#0a0a0a"
C_PANEL   = "#181818"
C_PANEL_HOVER = "#1f1f1f"
C_GREEN   = "#00e676"
C_CYAN    = "#00d4e8"
C_BLUE    = "#2196f3"
C_WHITE   = "#f5f5f5"
C_GRAY    = "#8a8a8a"
C_GRAY_DIM= "#5a5a5a"
C_DARK    = "#252525"
C_BORDER  = "#333333"
SPACE_XS = 6
SPACE_S = 10
SPACE_M = 16
SPACE_L = 20

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
    settings_applied = pyqtSignal(float, int, int, str, int, int, str, bool, bool, bool)

    def __init__(self, parent=None, current_camera=0, initial_settings=None):
        super().__init__(parent)
        self.current_camera = current_camera
        self.initial_settings = initial_settings or {}
        self.setWindowFlags(Qt.Widget)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedSize(560, 700)
        self.setStyleSheet(f"background: #141414; border-left: 3px solid {C_GREEN};")
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        layout.setSpacing(SPACE_M)

        # Header
        header_row = QHBoxLayout()
        title = QLabel(t("settings_title"))
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
        close_btn.clicked.connect(lambda: self.parent()._toggle_settings())
        header_row.addWidget(close_btn)
        layout.addLayout(header_row)

        self._divider(layout)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none; background: transparent;
            }}
            QTabBar::tab {{
                background: {C_DARK}; color: {C_GRAY};
                padding: 8px 14px; margin-right: 4px;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
                font-family: 'Courier New'; font-size: 11px;
            }}
            QTabBar::tab:selected {{
                background: {C_GREEN}; color: #000; font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{
                color: {C_WHITE};
            }}
        """)

        general_tab = QWidget()
        display_tab = QWidget()
        sound_tab   = QWidget()
        about_tab   = QWidget()

        self.tabs.addTab(general_tab, t("tab_general"))
        self.tabs.addTab(display_tab, t("tab_display"))
        self.tabs.addTab(sound_tab, t("tab_sound"))
        self.tabs.addTab(about_tab, t("tab_about"))

        layout.addWidget(self.tabs, 1)

        self._build_general_tab(general_tab)
        self._build_display_tab(display_tab)
        self._build_sound_tab(sound_tab)
        self._build_about_tab(about_tab)

        # Apply button
        apply_btn = QPushButton(t("apply_settings"))
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
        for text, slot in [(t("reset"), self._reset), (t("close"), self.hide)]:
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
        lbl.setFont(QFont("Courier New", 10, QFont.DemiBold))
        lbl.setStyleSheet(f"color: {C_GRAY}; border: none; letter-spacing: 1px;")
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
        language = self.lang_combo.currentData()
        window_fullscreen = self.get_window_fullscreen()
        mirror_preview = self.get_mirror_preview()
        show_fps = self.get_show_fps()

        self._model_path = getattr(self, '_model_path', f"models/{self.model_input.text()}")
        self.settings_applied.emit(confidence, hold, buffer, self._model_path,
                                camera_index, suggestion_count, language,
                                window_fullscreen, mirror_preview, show_fps)
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
        self.mute_btn.setText(t("sound_off") if checked else t("sound_on"))

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
        self.autostart_camera_btn.setText(f"{t('autostart_camera')}: {'ON' if checked else 'OFF'}")

    def _toggle_fullscreen_label(self, checked):
        self.fullscreen_btn.setText(f"{t('start_fullscreen')}: {'ON' if checked else 'OFF'}")

    def get_autostart_camera(self):
        return self.autostart_camera_btn.isChecked()

    def get_start_fullscreen(self):
        return self.fullscreen_btn.isChecked()

    def get_suggestion_count(self):
        return self.sugg_slider.value()

    def _build_general_tab(self, tab):
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, SPACE_M, 4, 4)
        layout.setSpacing(SPACE_M)

        layout.addWidget(self._label(t("language")))
        self.lang_combo = QComboBox()
        self.lang_combo.setFixedHeight(34)
        self.lang_combo.setStyleSheet(self._combo_style())
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("Azərbaycanca", "az")
        self.lang_combo.addItem("Русский", "ru")
        current = get_language()
        idx = {"en": 0, "az": 1, "ru": 2}.get(current, 0)
        self.lang_combo.setCurrentIndex(idx)
        layout.addWidget(self.lang_combo)

        self._divider(layout)

        layout.addWidget(self._label(t("model_file")))
        model_row = QHBoxLayout()
        self.model_input = QLabel("az_model.pkl")
        self.model_input.setFont(QFont("Courier New", 11))
        self.model_input.setStyleSheet(f"background: #1c1c1c; color: {C_WHITE}; border-radius: 6px; padding: 6px 10px; border: none;")
        self.model_input.setFixedHeight(34)
        browse_btn = QPushButton("...")
        browse_btn.setFixedSize(34, 34)
        browse_btn.setStyleSheet(f"""
            QPushButton {{ background: #1c1c1c; color: {C_GREEN}; border: none; border-radius: 6px; font-family: 'Courier New'; font-size: 14px; }}
            QPushButton:hover {{ background: #2a2a2a; }}
        """)
        browse_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self.model_input, 1)
        model_row.addWidget(browse_btn)
        layout.addLayout(model_row)

        self._divider(layout)

        layout.addWidget(self._label(t("camera")))
        self.camera_combo = QComboBox()
        self.camera_combo.setFixedHeight(34)
        self.camera_combo.setStyleSheet(self._combo_style())
        self._populate_cameras()
        layout.addWidget(self.camera_combo)

        refresh_cam_btn = QPushButton(t("refresh_cameras"))
        refresh_cam_btn.setFixedHeight(30)
        refresh_cam_btn.setStyleSheet(f"""
            QPushButton {{ background: #1c1c1c; color: {C_GRAY}; border: none; border-radius: 6px; font-family: 'Courier New'; font-size: 10px; }}
            QPushButton:hover {{ color: {C_WHITE}; background: #2a2a2a; }}
        """)
        refresh_cam_btn.clicked.connect(self._populate_cameras)
        layout.addWidget(refresh_cam_btn)

        self._divider(layout)

        conf_row = QHBoxLayout()
        conf_row.addWidget(self._label(t("confidence_threshold")))
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

        self._divider(layout)

        hold_row = QHBoxLayout()
        hold_row.addWidget(self._label(t("hold_speed")))
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

        self._divider(layout)

        buf_row = QHBoxLayout()
        buf_row.addWidget(self._label(t("smoothing_buffer")))
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

        self._divider(layout)

        sugg_row = QHBoxLayout()
        sugg_row.addWidget(self._label(t("suggestion_count")))
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

        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)


    def _build_display_tab(self, tab):
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, SPACE_M, 4, 4)
        layout.setSpacing(SPACE_M)

        layout.addWidget(self._label(t("startup_behavior")))

        self.autostart_camera_btn = QPushButton(f"{t('autostart_camera')}: ON")
        self.autostart_camera_btn.setFixedHeight(36)
        self.autostart_camera_btn.setCheckable(True)
        self.autostart_camera_btn.setChecked(True)
        self.autostart_camera_btn.setStyleSheet(self._toggle_style())
        self.autostart_camera_btn.clicked.connect(self._toggle_autostart_label)
        layout.addWidget(self.autostart_camera_btn)

        self.fullscreen_btn = QPushButton(f"{t('start_fullscreen')}: ON")
        self.fullscreen_btn.setFixedHeight(36)
        self.fullscreen_btn.setCheckable(True)
        self.fullscreen_btn.setChecked(True)
        self.fullscreen_btn.setStyleSheet(self._toggle_style())
        self.fullscreen_btn.clicked.connect(self._toggle_fullscreen_label)
        layout.addWidget(self.fullscreen_btn)

        self._divider(layout)

        self.window_mode_btn = QPushButton(f"{t('window_mode')}: {t('fullscreen')}")
        self.window_mode_btn.setFixedHeight(36)
        self.window_mode_btn.setCheckable(True)
        self.window_mode_btn.setChecked(True)
        self.window_mode_btn.setStyleSheet(self._toggle_style())
        self.window_mode_btn.clicked.connect(self._toggle_window_mode_label)
        layout.addWidget(self.window_mode_btn)

        self.mirror_btn = QPushButton(f"{t('mirror_preview')}: ON")
        self.mirror_btn.setFixedHeight(36)
        self.mirror_btn.setCheckable(True)
        self.mirror_btn.setChecked(True)
        self.mirror_btn.setStyleSheet(self._toggle_style())
        self.mirror_btn.clicked.connect(self._toggle_mirror_label)
        layout.addWidget(self.mirror_btn)

        self.fps_btn = QPushButton(f"{t('fps_counter')}: OFF")
        self.fps_btn.setFixedHeight(36)
        self.fps_btn.setCheckable(True)
        self.fps_btn.setChecked(False)
        self.fps_btn.setStyleSheet(self._toggle_style())
        self.fps_btn.clicked.connect(self._toggle_fps_label)
        layout.addWidget(self.fps_btn)

        layout.addStretch()


    def _build_sound_tab(self, tab):
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, SPACE_M, 4, 4)
        layout.setSpacing(SPACE_M)

        sound_row = QHBoxLayout()
        sound_row.addWidget(self._label(t("sound_volume")))
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

        self.mute_btn = QPushButton(t("sound_on"))
        self.mute_btn.setFixedHeight(36)
        self.mute_btn.setCheckable(True)
        self.mute_btn.setStyleSheet(f"""
            QPushButton {{ background: {C_DARK}; color: {C_WHITE}; border: none; border-radius: 6px; font-family: 'Courier New'; font-size: 12px; }}
            QPushButton:checked {{ background: #ff5f57; color: #000; }}
        """)
        self.mute_btn.clicked.connect(self._toggle_mute)
        layout.addWidget(self.mute_btn)

        layout.addStretch()


    def _build_about_tab(self, tab):
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, SPACE_M, 4, 4)
        layout.setSpacing(SPACE_S)
        layout.setAlignment(Qt.AlignTop)

        app_name = QLabel(t("app_title"))
        app_name.setFont(QFont("Courier New", 18, QFont.Bold))
        app_name.setStyleSheet(f"color: {C_GREEN}; border: none;")
        layout.addWidget(app_name)

        version = QLabel(t("version_label") + " 1.0.0")
        version.setFont(QFont("Courier New", 10))
        version.setStyleSheet(f"color: {C_GRAY}; border: none;")
        layout.addWidget(version)

        layout.addSpacing(SPACE_M)

        desc = QLabel(t("about_description"))
        desc.setFont(QFont("Courier New", 10))
        desc.setStyleSheet(f"color: {C_WHITE}; border: none;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(SPACE_M)

        author = QLabel(t("author_label") + " Elchin Huseynov")
        author.setFont(QFont("Courier New", 10))
        author.setStyleSheet(f"color: {C_GRAY}; border: none;")
        layout.addWidget(author)

        dataset = QLabel(t("dataset_credit"))
        dataset.setFont(QFont("Courier New", 9))
        dataset.setStyleSheet(f"color: {C_GRAY_DIM}; border: none;")
        dataset.setWordWrap(True)
        layout.addWidget(dataset)

        layout.addStretch()


    def _combo_style(self):
        return f"""
            QComboBox {{
                background: #1c1c1c; color: {C_WHITE};
                border-radius: 6px; padding: 6px 10px; border: 1px solid transparent;
                font-family: 'Courier New'; font-size: 12px;
            }}
            QComboBox:hover {{ border: 1px solid {C_GREEN}; }}
            QComboBox QAbstractItemView {{
                background: #1c1c1c; color: {C_WHITE};
                selection-background-color: {C_GREEN}; selection-color: #000;
            }}
        """

    def _toggle_style(self):
        return f"""
            QPushButton {{
                background: {C_GREEN}; color: #000;
                border: none; border-radius: 6px;
                font-family: 'Courier New'; font-size: 12px;
            }}
            QPushButton:!checked {{ background: {C_DARK}; color: {C_WHITE}; }}
        """

    def _toggle_window_mode_label(self, checked):
        mode = t("fullscreen") if checked else t("windowed")
        self.window_mode_btn.setText(f"{t('window_mode')}: {mode}")

    def _toggle_mirror_label(self, checked):
        self.mirror_btn.setText(f"{t('mirror_preview')}: {'ON' if checked else 'OFF'}")

    def _toggle_fps_label(self, checked):
        self.fps_btn.setText(f"{t('fps_counter')}: {'ON' if checked else 'OFF'}")

    def get_window_fullscreen(self):
        return self.window_mode_btn.isChecked()

    def get_mirror_preview(self):
        return self.mirror_btn.isChecked()

    def get_show_fps(self):
        return self.fps_btn.isChecked()

class Backdrop(QWidget):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: rgba(0, 0, 0, 150);")
        self.hide()

    def mousePressEvent(self, event):
        self.clicked.emit()


class MainWindow(QMainWindow):
    def __init__(self, model=None):
        super().__init__()
        self.settings = load_settings()

        self.model = model if model is not None else joblib.load(MODEL_FILE)
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.current_word = []
        self.sentence = []
        self.letter_history = deque(maxlen=HISTORY_SIZE)
        self.last_prediction = ""
        self.last_confidence = 0.0
        self.last_hold = 0

        set_language(self.settings["language"])

        global CONFIDENCE_MIN, HOLD_FRAMES, BUFFER_SIZE, SUGGESTION_COUNT
        CONFIDENCE_MIN = self.settings["confidence"]
        HOLD_FRAMES = self.settings["hold_frames"]
        BUFFER_SIZE = self.settings["buffer_size"]
        SUGGESTION_COUNT = self.settings["suggestion_count"]

        import sounds
        sounds.set_volume(self.settings["sound_volume"] / 100.0)
        sounds.set_muted(self.settings["sound_muted"])

        self.setWindowTitle("AzSL Recognition")
        self.showFullScreen()
        self.setStyleSheet(f"background-color: {C_BG}; color: {C_WHITE};")

        self._build_ui()
        self._setup_shortcuts()

        self.backdrop = Backdrop(self)
        self.backdrop.clicked.connect(self._close_any_modal)

        self.settings_panel = SettingsPanel(self, current_camera=self.settings["camera_index"])
        self.settings_panel.settings_applied.connect(self._apply_settings)
        self.settings_panel.hide()

        self.camera_thread = CameraThread(self.model, camera_index=self.settings["camera_index"])
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.prediction_ready.connect(self.update_prediction)
        self.camera_thread.start()

        # Stats panel
        self.stats_panel = StatsPanel(self)
        self.stats_panel.hide()
        self._reposition_stats()

        self.autocomplete = AutocompleteEngine()

    def _close_any_modal(self):
        if self.settings_panel.isVisible():
            self._toggle_settings
        if self.stats_panel.isVisible():
            self._toggle_stats()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(SPACE_M, SPACE_M, SPACE_M, SPACE_M)
        root.setSpacing(SPACE_S)

        # Title bar
        title_bar = QHBoxLayout()
        title = QLabel(t("app_title"))
        title.setFont(QFont("Courier New", 14))
        title.setStyleSheet(f"color: {C_GRAY};")
        title_bar.addWidget(title)
        title_bar.addStretch()

        # stats button
        stats_btn = QPushButton(t("stats"))
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
        settings_btn = QPushButton(t("settings"))
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

        quit_btn = QPushButton(t("quit"))
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
        main_row.setSpacing(SPACE_S + 2)

        self.cam_label = QLabel()
        self.cam_label.setMinimumSize(700, 480)
        self.cam_label.setStyleSheet(f"""
            background: {C_PANEL};
            border-radius: 12px;
            border: 1px solid {C_BORDER};
        """)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_row.addWidget(self.cam_label, 65)

        right = QVBoxLayout()
        right.setSpacing(SPACE_S)

        # Letter box
        letter_panel = QWidget()
        letter_panel.setMinimumHeight(220)
        letter_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 12px; border: 1px solid {C_BORDER};")
        letter_layout = QVBoxLayout(letter_panel)
        letter_layout.setContentsMargins(SPACE_M, SPACE_S, SPACE_M, SPACE_S)
        lbl_title = QLabel(t("detected_letter"))
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
        conf_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 12px; border: 1px solid {C_BORDER};")
        conf_layout = QVBoxLayout(conf_panel)
        conf_layout.setContentsMargins(SPACE_M, SPACE_S, SPACE_M, SPACE_S)
        conf_lbl = QLabel(t("confidence"))
        conf_lbl.setFont(QFont("Courier New", 10))
        conf_lbl.setStyleSheet(f"color: {C_GRAY}; border: none; letter-spacing: 1.5px; font-weight: 600;")
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setTextVisible(True)
        self.conf_bar.setFixedHeight(28)
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C_DARK}; border-radius: 6px; border: 1px solid {C_BORDER};
                color: {C_WHITE}; font-family: 'Courier New'; font-size: 12px; font-weight: bold;
                text-align: center;
            }}
            QProgressBar::chunk {{ background: {C_GREEN}; border-radius: 6px; }}
        """)
        conf_layout.addWidget(conf_lbl)
        conf_layout.addWidget(self.conf_bar)
        right.addWidget(conf_panel)

        # Hold bar
        hold_panel = QWidget()
        hold_panel.setMinimumHeight(80)
        hold_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 12px; border: 1px solid {C_BORDER};")
        hold_layout = QVBoxLayout(hold_panel)
        hold_layout.setContentsMargins(SPACE_M, SPACE_S, SPACE_M, SPACE_S)
        hold_lbl = QLabel(t("hold_progress"))
        hold_lbl.setFont(QFont("Courier New", 10))
        hold_lbl.setStyleSheet(f"color: {C_GRAY}; border: none; letter-spacing: 1.5px; font-weight: 600;")
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
        hist_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 12px; border: 1px solid {C_BORDER};")
        hist_layout = QVBoxLayout(hist_panel)
        hist_layout.setContentsMargins(SPACE_M, SPACE_S, SPACE_M, SPACE_S)
        hist_lbl = QLabel(t("history"))
        hist_lbl.setFont(QFont("Courier New", 10))
        hist_lbl.setStyleSheet(f"color: {C_GRAY}; border: none; letter-spacing: 1.5px; font-weight: 600;")
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
        word_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 10px; border: 1px solid {C_BORDER};")
        word_row = QHBoxLayout(word_panel)
        word_row.setContentsMargins(SPACE_L, 0, SPACE_L, 0)
        word_lbl = QLabel(t("word"))
        word_lbl.setFixedWidth(90)
        word_lbl.setFont(QFont("Courier New", 10))
        word_lbl.setStyleSheet(f"color: {C_GRAY}; border: none; letter-spacing: 1.5px; font-weight: 600;")
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
        self.suggestions_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 10px; border: 1px solid {C_BORDER};")
        sugg_layout = QHBoxLayout(self.suggestions_panel)
        sugg_layout.setContentsMargins(SPACE_M, 0, SPACE_M, 0)
        sugg_layout.setSpacing(SPACE_S)

        sugg_label = QLabel(t("suggest"))
        sugg_label.setFixedWidth(90)
        sugg_label.setFont(QFont("Courier New", 10))
        sugg_label.setStyleSheet(f"color: {C_GRAY}; border: none; letter-spacing: 1.5px; font-weight: 600;")
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
        sent_row.setSpacing(SPACE_S)
        sent_panel = QWidget()
        sent_panel.setFixedHeight(64)
        sent_panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 10px; border: 1px solid {C_BORDER};")
        sent_layout = QHBoxLayout(sent_panel)
        sent_layout.setContentsMargins(SPACE_L, 0, SPACE_L, 0)
        sent_lbl = QLabel(t("sentence"))
        sent_lbl.setFixedWidth(90)
        sent_lbl.setFont(QFont("Courier New", 10))
        sent_lbl.setStyleSheet(f"color: {C_GRAY}; border: none; letter-spacing: 1.5px; font-weight: 600;")
        self.sent_label = QLabel("...")
        self.sent_label.setFont(QFont("Courier New", 18))
        self.sent_label.setStyleSheet(f"color: {C_BLUE}; border: none;")
        sent_layout.addWidget(sent_lbl)
        sent_layout.addWidget(self.sent_label)
        sent_layout.addStretch()
        sent_row.addWidget(sent_panel, 1)

        speak_btn = QPushButton(t("speak"))
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

        export_btn = QPushButton(t("export"))
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
        controls = QLabel(t("controls_hint"))
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
            self._animate_model_out(self.settings_panel)
            self.backdrop.hide()
        else:
            self._center_panel(self.settings_panel)
            self.backdrop.setGeometry(0, 0, self.width(), self.height())
            self.backdrop.show()
            self.backdrop.raise_()
            self.settings_panel.raise_()
            self.settings_panel.show()
            self._animate_modal_in(self.settings_panel)
    
    def _toggle_stats(self):
        if self.stats_panel.isVisible():
            self._animate_panel_out(self.stats_panel)
        else:
            self._reposition_stats()
            panel_w = self.stats_panel.width()
            start_x = self.width()
            end_x = self.width() - panel_w
            self.stats_panel.move(start_x, 0)
            self.stats_panel.raise_()
            self.stats_panel.show()
            self._animate_panel_in(self.stats_panel, end_x)

    def _center_panel(self, panel):
        x = (self.width() - panel.width()) // 2
        y = (self.height() - panel.height()) // 2
        panel.move(x, y)

    def _animate_modal_in(self, panel):
        panel.setWindowOpacity(0.0)
        anim = QPropertyAnimation(panel, b"windowOpacity")
        anim.setDuration(180)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        panel._anim = anim

    def _animate_model_out(self, panel):
        anim = QPropertyAnimation(panel, b"windowOpacity")
        anim.setDuration(140)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.InCubic)
        anim.finished.connect(panel.hide)
        anim.start()
        panel._anim = anim
          
    def _animate_panel_in(self, panel, end_x):
        anim = QPropertyAnimation(panel, b"pos")
        anim.setDuration(220)
        anim.setStartValue(panel.pos())
        anim.setEndValue(panel.pos().__class__(end_x, panel.pos().y()))
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        panel._anim = anim

    def _animate_panel_out(self, panel):
        end_x = self.width()
        anim = QPropertyAnimation(panel, b"pos")
        anim.setDuration(180)
        anim.setStartValue(panel.pos())
        anim.setEndValue(panel.pos().__class__(end_x, panel.pos().y()))
        anim.setEasingCurve(QEasingCurve.InCubic)
        anim.finished.connect(panel.hide)
        anim.start()
        panel._anim = anim

    def _pulse_widget(self, widget):
        from PyQt5.QtCore import QRect
        anim = QPropertyAnimation(widget, b"geometry")
        anim.setDuration(150)
        rect = widget.geometry()
        grown = QRect(rect.x() - 4, rect.y() - 4, rect.width() + 8, rect.height() +8)
        anim.setKeyValueAt(0, rect)
        anim.setKeyValueAt(0.5, grown)
        anim.setKeyValueAt(1, rect)
        anim.setEasingCurve(QEasingCurve.OutQuad)
        anim.start()
        widget._pulse_anim = anim

    def _animate_bar(self, bar, target_value):
        anim = QPropertyAnimation(bar, b"value")
        anim.setDuration(120)
        anim.setStartValue(bar.value())
        anim.setEndValue(target_value)
        anim.setEasingCurve(QEasingCurve.OutQuad)
        anim.start()
        bar._anim = anim

    def _reposition_stats(self):
        panel_w = 300
        self.stats_panel.setGeometry(self.width() - panel_w, 0, panel_w, self.height())


    def _apply_settings(self, confidence, hold, buffer, model_path, camera_index,
                    suggestion_count, language, window_fullscreen, mirror_preview, show_fps):
        global CONFIDENCE_MIN, HOLD_FRAMES, BUFFER_SIZE, SUGGESTION_COUNT
        CONFIDENCE_MIN = confidence
        HOLD_FRAMES = hold
        BUFFER_SIZE = buffer
        SUGGESTION_COUNT= suggestion_count

        if language != get_language():
            set_language(language)
            self._rebuild_ui_language()

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

        if window_fullscreen:
            self.showFullScreen()
        else:
            self.showNormal()

        self.mirror_preview = mirror_preview
        self.show_fps = show_fps
        
        self.settings.update({
            "confidence": confidence,
            "hold_frames": hold,
            "buffer_size": buffer,
            "camera_index": camera_index,
            "suggestion_count": suggestion_count,
            "language": language,
            "model_path": model_path,
            "sound_volume": self.settings_panel.sound_slider.value(),
            "sound_muted": self.settings_panel.mute_btn.isChecked(),
            "autostart_camera": self.settings_panel.get_autostart_camera(),
            "start_fullscreen": self.settings_panel.get_start_fullscreen(),
            "window_fullscreen": window_fullscreen,
            "mirror_preview": mirror_preview,
            "show_fps": show_fps,
        })
        save_settings(self.settings)

        print(f"Settings applied: conf={confidence} hold={hold} buffer={buffer} camera={camera_index}")

    def _rebuild_ui_language(self):
        old_central = self.centralWidget()
        old_central.deleteLater()
        self._build_ui()
        self._refresh_display()

        old_camera = self.settings_panel.current_camera
        self.settings_panel.deleteLater()
        self.settings_panel = SettingsPanel(self, current_camera=old_camera, initial_settings=self.settings)
        self.settings_panel.settings_applied.connect(self._apply_settings)
        self.settings_panel.hide()

        old_letter_counts = self.stats_panel.letter_counts.copy()
        old_total_letters = self.stats_panel.total_letters
        old_total_words = self.stats_panel.total_words
        old_session_start = self.stats_panel.session_start

        self.stats_panel.deleteLater()
        self.stats_panel = StatsPanel(self)
        self.stats_panel.letter_counts = old_letter_counts
        self.stats_panel.total_letters = old_total_letters
        self.stats_panel.total_words = old_total_words
        self.stats_panel.session_start = old_session_start
        self.stats_panel._refresh_bars()
        self.stats_panel._refresh_overview()
        self.stats_panel.hide()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'backdrop') and self.backdrop.isVisible():
            self.backdrop.setGeometry(0, 0, self.width(), self.height())
        if hasattr(self, 'settings_panel') and self.settings_panel.isVisible():
            self._center_panel(self.settings_panel)
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
            self.letter_label.setStyleSheet(f"color: {C_GREEN}; border: none;")
            conf_pct  = int(confidence * 100)
            self._animate_bar(self.conf_bar, conf_pct)
            self.conf_bar.setFormat(f"{conf_pct}%")
            color = C_GREEN if confidence > 0.75 else C_BLUE
            self.conf_bar.setStyleSheet(f"""
                QProgressBar {{
                    background: {C_DARK}; border-radius: 6px; border: 1px solid {C_BORDER};
                    color: {C_WHITE}; font-family: 'Courier New'; font-size: 12px; font-weight: bold;
                    text-align: center;
                }}
                QProgressBar::chunk {{ background: {color}; border-radius: 6px; }}
            """)
            self._animate_bar(self.hold_bar, hold_counter)

            if hold_counter >= HOLD_FRAMES:
                self.current_word.append(prediction)
                self.letter_history.append(prediction)
                self.camera_thread.confirm_letter(prediction)
                sounds.play_letter_confirm()
                self.stats_panel.record_letter(prediction)
                self._refresh_display()
                self._pulse_widget(self.letter_label.parentWidget())
        else:
            self.letter_label.setText("—")
            self.letter_label.setStyleSheet(f"color: {C_GRAY_DIM}; border: none;")
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
        suggestions = self.autocomplete.suggest(prefix, max_results=SUGGESTION_COUNT)
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