import cv2
import csv
import os
from mediapipe.python.solutions import hands as mp_hands_module
from utils import extract_features

# ── Config ─────────────────────────────────────────────
IMAGE_DATASET_PATH = r"E:\1. Private\Coding\MediaPipe_Processed_ASL_Dataset\processed_combine_asl_dataset"
OUTPUT_CSV         = "data/asl_data.csv"
SKIP_LABELS        = {"nothing", "space", "del"}
# ───────────────────────────────────────────────────────

os.makedirs("data", exist_ok=True)

with mp_hands_module.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:
    
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        labels = sorted(os.listdir(IMAGE_DATASET_PATH))
        total = len(labels)

        for i, label in images:
            img_path = os.path.join(label_path, img_file)
            image    = cv2.imread(img_path)
            if image is None:
                continue

            rgb      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

"""
will continue tomorrow.......
"""


print(f"\nAll done! CSV saved to: {OUTPUT_CSV}")
