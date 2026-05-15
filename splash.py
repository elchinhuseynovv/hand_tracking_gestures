import sys
import joblib
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QProgressBar, QVBoxLayout
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

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(520, 340)
        self._center()
        self._build_ui()
        self._start_loading()

    def _center(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def _build_ui(self):
        container = QWidget(self)
        container.setGeometry(0, 0, 520, 340)
        container.setStyleSheet(f"""
            QWidget {{
                background: {C_BG};
                border-radius: 16px;
                border: 1px solid #2a2a2a;
            }}
        """)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(50, 50, 50, 40)
        layout.setSpacing(0)

        # Logo - will be placeholder now
        logo = QLabel("✋")
        logo.setFont(QFont("Segoe UI Emoji", 52))
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(logo)

        layout.addSpacing(12)


        # App title
        title = QLabel("AzSL Recognition")
        title.setFont(QFont("Courier New", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {C_GREEN}; border: none; background: transparent;")
        layout.addWidget(title)

        layout.addSpacing(4)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: #2a2a2a;
                border-radius: 2px;
                border: none;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {C_GREEN}, stop:1 {C_CYAN}
                );
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        layout.addSpacing(12)


        # Status label
        self.status_label = QLabel("Starting up...")
        self.status_label.setFont(QFont("Courier New", 10))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"color: {C_GRAY}; border: none; background: transparent;")
        layout.addWidget(self.status_label)

        layout.addSpacing(20)

        # Version
        version = QLabel("v1.0.0")
        version.setFont(QFont("Courier New", 9))
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet(f"color: #333; border: none; background: transparent;")
        layout.addWidget(version)

        
    def _start_loading(self):
        self.loader = LoaderThread()
        self.loader.progress.connect(self._update_progress)
        self.loader.finished.connect(self._on_loaded)
        self.loader.start()

    def _update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            f"color: {C_GREEN if value == 100 else C_GRAY}: border: none; background: transparent;"
        )
        
    def _on_loaded(self, model):
        self.main_window = MainWindow(model)
        self.main_window.show()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    splash = SplashScreen()
    splash.show()
    sys.exit(app.exec())
