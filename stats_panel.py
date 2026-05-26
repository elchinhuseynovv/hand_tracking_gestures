from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from collections import defaultdict
import time

C_BG    = "#0f0f0f"
C_PANEL = "#1c1c1c"
C_GREEN = "#00dc64"
C_CYAN  = "#00c8dc"
C_BLUE  = "#00a5ff"
C_WHITE = "#ffffff"
C_GRAY  = "#555555"
C_DARK  = "#2a2a2a"
C_AMBER = "#ffb347"


class StatsPanel(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Widget)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"background: #141414; border-left: 3px solid {C_CYAN};")

        # Session data
        self.session_start = time.time()
        self.letter_counts = defaultdict(int)
        self.letter_counts    = defaultdict(int)   # how many times each letter confirmed
        self.prediction_hits  = defaultdict(int)   # correct predictions per letter
        self.prediction_total = defaultdict(int)   # total predictions per letter
        self.total_letters = 0
        self.total_words = 0

        self._build()

        # Auto refresh every second
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh_time)
        self.timer.start(1000)

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing()

        # Header
        header_row = QHBoxLayout()
        title = QLabel("STATISTICS")
        title.setFont(QFont("Courier New", 14, QFont.Bold))
        title.setStyleSheet(f"color: {C_CYAN}; border: none;")
        header_row.addWidget(title)
        header_row.addStretch()
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(32, 26)
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

        # Session overview

        overview_lbl = QLabel("SESSION OVERVIEW")
        overview_lbl.setFont(QFont("Courier New", 10))
        overview_lbl.setStyleSheet(f"color: {C_GRAY}; border: none;")
        layout.addWidget(overview_lbl)

        stats_grid = QHBoxLayout()
        self.time_box    = self._stat_box("00:00", "TIME")
        self.letters_box = self._stat_box("0", "LETTERS")
        self.words_box   = self._stat_box("0", "WORDS")
        stats_grid.addWidget(self.time_box)
        stats_grid.addWidget(self.letters_box)
        stats_grid.addWidget(self.words_box)
        layout.addLayout(stats_grid)

        self._divider(layout)

        # Per-letter breakdown\
        breakdown_lbl = QLabel("LETTER BREAKDOWN")
        breakdown_lbl.setFont(QFont("Courier New", 10))
        breakdown_lbl.setStyleSheet(f"color: {C_GRAY}; border: none;")
        layout.addWidget(breakdown_lbl)

        # Scrollable area for letter bars
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                background: #1c1c1c; width: 6px; border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #3a3a3a; border-radius: 3px;
            }
        """)

        self.bars_widget = QWidget()
        self.bars_widget.setStyleSheet("background: transparent;")
        self.bars_layout = QVBoxLayout(self.bars_widget)
        self.bars_layout.setSpacing(6)
        self.bars_layout.setContentsMargins(0, 0, 0, 0)

        # Create a bar for each letter
        self.letter_bars = {}
        letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        for letter in letters:
            row = QHBoxLayout()
            lbl = QLabel(letter)
            lbl.setFixedWidth(24)
            lbl.setFont(QFont("Courier New", 11, QFont.Bold))
            lbl.setStyleSheet(f"color: {C_WHITE}; border: none;")

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(14)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background: {C_DARK};
                    border-radius: 4px;
                    border: none;
                }}
                QProgressBar::chunk {{
                    background: {C_GREEN};
                    border-radius: 4px;
                }}
            """)

            count_lbl = QLabel("0")
            count_lbl.setFixedWidth(32)
            count_lbl.setFont(QFont("Courier New", 9))
            count_lbl.setAlignment(Qt.AlignRight)
            count_lbl.setStyleSheet(f"color: {C_GRAY}; border: none;")

            row.addWidget(lbl)
            row.addWidget(bar, 1)
            row.addWidget(count_lbl)
            self.bars_layout.addLayout(row)
            self.letter_bars[letter] = (bar, count_lbl)

        self.bars_layout.addStretch()
        scroll.setWidget(self.bars_widget)
        layout.addWidget(scroll, 1)

        self._divider(layout)

        # ── Reset button ────────────────────────────────
        reset_btn = QPushButton("RESET STATS")
        reset_btn.setFixedHeight(36)
        reset_btn.setStyleSheet(f"""
            QPushButton {{
                background: #1c1c1c; color: {C_GRAY};
                border: none; border-radius: 6px;
                font-family: 'Courier New'; font-size: 11px;
            }}
            QPushButton:hover {{ color: {C_WHITE}; background: #2a2a2a; }}
        """)
        reset_btn.clicked.connect(self._reset_stats)
        layout.addWidget(reset_btn)

    def _stat_box(self, value, label):
        box = QWidget()
        box.setStyleSheet(f"""
            background: {C_PANEL};
            border-radius: 8px;
            border: 1px solid #2a2a2a;
        """)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(10, 8, 10, 8)
        box_layout.setSpacing(2)

        val_lbl = QLabel(value)
        val_lbl.setFont(QFont("Courier New", 18, QFont.Bold))
        val_lbl.setStyleSheet(f"color: {C_CYAN}; border: none;")
        val_lbl.setAlignment(Qt.AlignCenter)

        box_layout.addWidget(val_lbl)
        box_layout.addWidget(key_lbl)
        box._value_label = val_lbl
        return box
    
    def _divider(self, layout);
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background: #2a2a2a; border: none;")
        layout.addWidget(line)

    # Public methods called from MainWindow
    def record_letter(self, letter):
        self.letter_counts[letter] += 1
        self.total_letters += 1
        self._refresh_bars()
        self._refresh_overview()

    def record_word(self):
        self.total_words += 1
        self._refresh_overview()

    def record_prediction(self, predicted, actual=None)
        if predicted:
            self.prediction_total[predicted] += 1

    # Internal refresh methods
    def _refresh_bars(self):
        if not self.letter_counts:
            return
        max_count = max(self.letter_counts.values()) or 1
        for letter, (bar, count_lbl) in self.letter_bars,items():
            count = self.letter_counts.get(letter, 0)
            pct = int((count / max_count) * 100)
            bar.setValue(pct)
            count_lbl.setText(str(count))

            if pct > 66:
                color = C_GREEN
            elif pct > 33:
                color = C_CYAN
            else:
                color = C_BLUE
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background: {C_DARK};
                    border-radius: 4px; border: none;
                }}
                QProgressBar::chunk {{
                    background: {color};
                    border-radius: 4px;
                }}
            """)
        
        def _refresh_overview(self):
            self.letters_box._value_label.setText(str(self.total_letters))
            self.words_box._value_labe.setText(str(self.total_words))

        def _refresh_time(self):
            elapsed = int(time.time() - self.sessions_start)
            mins = elapsed // 60
            secs = elapsed % 60
            self.time_box._value_label.setText(f"{mins:02d}:{secs:02d}")

        def _reset_stats(self):
            self.session_start = time.time()
            self.letter_counts = defaultdict(int)
            self.prediction_total = defaultdict(int)
            self.total_letters = 0
            self.total_words = 0
            for letter, (bar, count_lbl) in self.letter_bars.items():
                bar.setValue(0)
                count_lbl.setText("0")
            self._refresh_overview()