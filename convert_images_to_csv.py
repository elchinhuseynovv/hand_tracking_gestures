import cv2
import csv
import os
from mediapipe.python.solutions import hands as mp_hands_module
from utils import extract_features

# ── Config ─────────────────────────────────────────────
IMAGE_DATASET_PATH = r"E:\1. Private\Coding\ASL_Data\asl_alphabet_train\asl_alphabet_train"
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
        total  = len(labels)

        for i, label in enumerate(labels):       # ← was missing enumerate
            label_clean = label.upper()          # ← these lines were missing
            if label_clean.lower() in SKIP_LABELS:
                continue

            label_path = os.path.join(IMAGE_DATASET_PATH, label)
            if not os.path.isdir(label_path):
                continue

            images        = os.listdir(label_path)
            success_count = 0                    # ← was missing

            for img_file in images:              # ← inner loop was missing
                img_path = os.path.join(label_path, img_file)
                image    = cv2.imread(img_path)
                if image is None:
                    continue

                rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    features = extract_features(results.multi_hand_landmarks[0])
                    writer.writerow(features + [label_clean])
                    success_count += 1

            print(f"[{i+1}/{total}] {label_clean}: {success_count}/{len(images)} images converted")

print(f"\nAll done! CSV saved to: {OUTPUT_CSV}")