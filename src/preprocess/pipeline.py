import cv2
import numpy as np
import os

# find project root dynamically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
TRAIN_IMAGES = os.path.join(DATA_PATH, "training_data/images")
TRAIN_ANNOTS = os.path.join(DATA_PATH, "training_data/annotations")


def preprocess_image(path):
    img = cv2.imread()
