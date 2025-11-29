import os 
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    """
    Loads images + labels from the local /data/dataset/ folder.
    Uses only os + pathlib, no fancy libraries.
    """

    def __init__(self, root_dir="data/dataset", transform=None):
     pass