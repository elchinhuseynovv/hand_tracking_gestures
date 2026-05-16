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

    # horizontal flip

    # blur variations

    # contrast adjustment

# counting exisiting samples



# boost each weak letter (# lower threshold to get more detections)




