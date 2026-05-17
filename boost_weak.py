import cv2
import csv
import os
import random
import numpy as np
from mediapipe.python.solutions import hands as mp_hands_module
from utils import extract_features

# Config
IMAGE_DATASET_PATH = r"E:\1. Private\Coding\AzSLD_Fingerspelling"
DATA_FILE          = "data/az_data.csv"
BOOST_TARGETS      = {
    'P': 500,   # boost P heavily — most confused
    'C': 400,   # boost C moderately
}

def augment_image(image):
    augmented = []
    h, w = image.shape[:2]

    for alpha in [0.5, 0.65, 0.8, 1.2, 1.4, 1.6]:
        bright = cv2.convertScaleAbs(image, alpha=alpha,
                                     beta=random.randint(-40, 40))
        augmented.append(bright)

    
    for angle in [-20, -12, -6, 6, 12, 20]:
        M  = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented.append(rotated)

    # zoom variations
    for scale in [0.75, 0.82, 0.90, 0.95]:
        new_w, new_h = int(w * scale), int(h * scale)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        cropped = image[y1:y1+new_h, x1:x1+new_w]
        zoomed = cv2.resize(cropped, (w, h))
        augmented.append(zoomed)
    # horizontal flip
    augmented.append(cv2.flip(image, 1))

    # blur variations
    for ksize in [3, 5]:
        augmented.append(cv2.GaussianBlur(image, (ksize, ksize), 0))

    # contrast adjustment
    for alpha in [0.7, 1.3]:
        contrast = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented.append(contrast)

    return augmented

# counting exisiting samples
print("Reading existing CSV...")
existing = {}
with open(DATA_FILE, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        label = row[-1]
        existing[label] = existing.get(label, 0) + 1

print("\nCurrent counts for weak letters:")
for letter in BOOST_TARGETS:
    print(f"  {letter}: {existing.get(letter, 0)} samples")

# boost each weak letter (# lower threshold to get more detections)
with mp_hands_module.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.4
) as hands:
    
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        for letter, target in BOOST_TARGETS.items():
            current = existing.get(letter, 0)
            needed = target - current

            if needed <= 0:
                print(f"\n{letter}: already at {current} - skipping")
                continue

            print(f"\nBoosting {letter}: {current} - {target} (need {needed} more)")

            label_path = os.path.join(IMAGE_DATASET_PATH, letter)
            if not os.path.isdir(label_path):
                print(f" Folder not found: {label_path}")
                continue

            images = [f for f in os.listdir(label_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                print(f"No images found!")
                continue
            
            added = 0
            attempts = 0

            while added < needed and attempts < needed * 30:
                attempts += 1
                img_file = random.choice(images)
                img_path = os.path.join(label_path, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                variants = augment_image(image)
                random.shuffle(variants)

                for variant in variants:
                    if added >= needed:
                        break
                    rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    if results.multi_hand_landmarks:
                        features = extract_features(results.multi_hand_landmarks[0])
                        writer.writerow(features + [letter])
                        added +=1

                print(f" Added {added}/{needed} samples")

print("\nDone! New counts:")
final = {}
with open(DATA_FILE, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        label = row[-1]
        final[label] = final.get(label, 0) + 1

for letter in BOOST_TARGETS:
    print(f"{letter} : {final.get(letter, 0)}")   

