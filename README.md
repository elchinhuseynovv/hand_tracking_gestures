# AzSL Recognition — Azerbaijani Sign Language Fingerspelling

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15-purple)
![Accuracy](https://img.shields.io/badge/Accuracy-95.98%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

A real-time Azerbaijani Sign Language (AzSL) fingerspelling recognition desktop application built with Python, MediaPipe, and PyQt5. The app detects hand landmarks through a webcam, classifies them using a trained Random Forest model, and builds words and sentences from confirmed signs — with text-to-speech output.

---

## Demo

> Sign a letter → hold it steady → the app confirms it → build words → press ENTER to speak

---

## Features

- **Real-time hand tracking** using MediaPipe (21 landmarks per hand)
- **95.98% classification accuracy** across 25 Azerbaijani/Latin letters
- **Word and sentence builder** — hold-to-confirm gesture input
- **Text-to-speech** output via pyttsx3
- **Polished dark UI** built with PyQt5 — fullscreen, responsive
- **Settings panel** — adjust confidence threshold, hold speed, and smoothing buffer live
- **Statistics panel** — per-letter usage breakdown, session timer, word count
- **Splash/loading screen** on startup
- **Data augmentation pipeline** — brightness, rotation, zoom, flip variations
- **Confusion analysis tools** — identify and boost weak letter classes

---

## Project Structure

```
asl_recognition/
│
├── app.py                    # Main PyQt5 application
├── splash.py                 # Splash/loading screen (entry point)
├── stats_panel.py            # Per-letter statistics panel
├── utils.py                  # Feature extraction (MediaPipe → 63-value vector)
│
├── 1_collect_data.py         # Webcam-based data collection script
├── convert_images_to_csv.py  # Convert image dataset → landmark CSV
├── augment_data.py           # Data augmentation for all classes
├── boost_weak.py             # Targeted augmentation for weak letters
├── analyze_confusion.py      # Confusion matrix analysis
├── 2_train_model.py          # Model training (Random Forest)
├── 3_realtime_recognition.py # Lightweight OpenCV-only recognition
│
├── data/
│   └── az_data.csv           # Extracted landmark features + labels
│
└── models/
    └── az_model.pkl          # Trained Random Forest classifier
```

---

## How It Works

Instead of classifying raw images (slow, GPU-heavy), the app extracts **63 normalized landmark coordinates** from MediaPipe's 21 hand joints and feeds them into a Random Forest classifier. All coordinates are relative to the wrist and normalized by hand scale — making recognition position and size independent.

```
Webcam → MediaPipe (21 landmarks) → Normalize → Random Forest → Prediction
```

---

## Model Performance

Trained on **7,213 samples** across 25 letters using the [AzSLD Fingerspelling Dataset](https://doi.org/10.5281/zenodo.14222948):

| Metric | Score |
|---|---|
| Overall Accuracy | **95.98%** |
| Best letters (B, D, R, X) | **100%** |
| Weakest letter (L) | **85%** |
| Training samples | 7,213 |
| Test samples | 1,443 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/elchinhuseynov/azsl-recognition.git
cd azsl-recognition

# Install dependencies
pip install opencv-python mediapipe scikit-learn PyQt5 pyttsx3 joblib numpy pandas
```

---

## Usage

```bash
# Launch the full app with splash screen
python splash.py

# Or run directly without splash
python app.py

# Collect your own training data
python 1_collect_data.py

# Retrain the model
python 2_train_model.py
```

### Controls

| Key | Action |
|---|---|
| Hold sign | Confirm letter (cyan bar fills up) |
| `SPACE` | Confirm current word |
| `ENTER` | Speak full sentence |
| `BACKSPACE` | Delete last letter |
| `C` | Clear everything |
| `ESC` | Quit |

---

## Dataset

This project uses the **AzSLD (Azerbaijani Sign Language Dataset)** — Fingerspelling subset:

- **Source:** [Zenodo — AzSLD](https://doi.org/10.5281/zenodo.14222948)
- **Content:** Real hand photos across 32 AzSL letters (24 static + 8 dynamic)
- **Used:** 25 static letters (dynamic letters require motion detection — planned for future)

---

## Roadmap

- [ ] Record and train the 7 special Azerbaijani letters (Ç, Ö, Ü, Ğ, İ, Ş, Ə)
- [ ] Dynamic letter support (movement-based signs)
- [ ] Package as standalone `.exe`
- [ ] Camera selection in settings
- [ ] Word prediction / autocomplete

---

## Tech Stack

| Library | Purpose |
|---|---|
| MediaPipe | Hand landmark detection |
| OpenCV | Webcam capture and image processing |
| scikit-learn | Random Forest classifier |
| PyQt5 | Desktop GUI |
| pyttsx3 | Text-to-speech |
| joblib | Model serialization |
| NumPy / Pandas | Data processing |

---

## Author

**Elchin Huseynov**
- GitHub: [@elchinhuseynov](https://github.com/elchinhuseynov)

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [AzSLD Dataset](https://doi.org/10.5281/zenodo.14222948) — Azerbaijani Sign Language Dataset
- [MediaPipe](https://mediapipe.dev/) — Google's hand tracking solution
- [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — Used during early development