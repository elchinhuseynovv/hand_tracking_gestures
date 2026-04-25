import cv2
import numpy as np
import csv
import os
from mediapipe.python.solutions import hands as mp_hands_module
from utils import extract_features
import random

# ── Config ─────────────────────────────────────────────
IMAGE_DATASET_PATH = r"E:\1. Private\Coding\AzSLD_Fingerspelling"
OUTPUT_CSV         = "data/az_data.csv"
TARGET_SAMPLES     = 150   # boost every class up to this number
# ───────────────────────────────────────────────────────

def augment_image(image):
    """Apply random augmentations to an image and return variations."""
    augmented = []

    # 1. Brightness change
    bright = cv2.convertScaleAbs(image, alpha=random.uniform(0.7, 1.3),
                                  beta=random.randint(-30, 30))
    augmented.append(bright)

    # 2. Horizontal flip (mirror)
    flipped = cv2.flip(image, 1)
    augmented.append(flipped)

    # 3. Slight rotation
    h, w = image.shape[:2]
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    augmented.append(rotated)

    # 4. Zoom in slightly
    scale  = random.uniform(0.85, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    cropped = image[y1:y1+new_h, x1:x1+new_w]
    zoomed  = cv2.resize(cropped, (w, h))
    augmented.append(zoomed)

    # 5. Gaussian blur (simulates motion/focus)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    augmented.append(blurred)

    return augmented

# ── Count existing samples per label ───────────────────
print("Reading existing CSV...")
existing = {}
with open(OUTPUT_CSV, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        label = row[-1]
        existing[label] = existing.get(label, 0) + 1

print("\nCurrent sample counts:")
for label, count in sorted(existing.items()):
    status = "✅" if count >= TARGET_SAMPLES else f"❌ needs {TARGET_SAMPLES - count} more"
    print(f"  {label}: {count} {status}")

# ── Augment weak classes ────────────────────────────────
print(f"\nBoosting all classes to {TARGET_SAMPLES} samples...")

with mp_hands_module.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)

        labels = sorted(os.listdir(IMAGE_DATASET_PATH))

        for label in labels:
            label_clean = label.upper()
            current_count = existing.get(label_clean, 0)

            if current_count >= TARGET_SAMPLES:
                print(f"  {label_clean}: already has {current_count} — skipping")
                continue

            needed = TARGET_SAMPLES - current_count
            label_path = os.path.join(IMAGE_DATASET_PATH, label)
            if not os.path.isdir(label_path):
                continue

            images = [f for f in os.listdir(label_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                continue

            added = 0
            attempts = 0
            max_attempts = needed * 20

            while added < needed and attempts < max_attempts:
                attempts += 1

                # Pick a random image from this label
                img_file = random.choice(images)
                img_path = os.path.join(label_path, img_file)
                image    = cv2.imread(img_path)
                if image is None:
                    continue

                # Apply augmentations
                variants = augment_image(image)
                random.shuffle(variants)

                for variant in variants:
                    if added >= needed:
                        break
                    rgb     = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        features = extract_features(results.multi_hand_landmarks[0])
                        writer.writerow(features + [label_clean])
                        added += 1

            print(f"  {label_clean}: added {added}/{needed} augmented samples")

print("\nDone! Recount:")
final = {}
with open(OUTPUT_CSV, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        label = row[-1]
        final[label] = final.get(label, 0) + 1

for label, count in sorted(final.items()):
    print(f"  {label}: {count}")