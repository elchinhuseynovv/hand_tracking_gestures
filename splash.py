import sys
import joblib
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QProgressBar, QBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor
from app import MainWindow

# COLORS

C_BG    = "#0f0f0f"
C_GREEN = "#00dc64"
C_CYAN  = "#00c8dc"
C_GRAY  = "#555555"
C_WHITE = "#ffffff"
C_DARK  = "#1c1c1c"

class LoaderThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)

    def run(self):
        import time

        self.progress.emit(30, "Initializing...")
        time.sleep(0.4)

        self.progress.emit(30, "Loadning MediaPipe...")
        from mediapipe.python.solutions import hands as mp_hands_module
        time.sleep(0.3)

        self.progress.emit(55, "Loading Model...")
        model = joblib.load("models/az_model.pkl")
        time.sleep(0.3)

        self.progress.emit(75, "Starting camera thred...")
        time.sleep(0.3)

        self.progress.emit(90, "Building interface...")
        time.sleep(0.4)

        self.progress.emit(100, "Ready!")
        time.sleep(0.3)

        self.finished.emit(model)

        