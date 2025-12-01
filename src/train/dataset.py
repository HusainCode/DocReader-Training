import os
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset


# Automatically detect project root even inside WSL or Docker
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class OCRDataset(Dataset):
    """
    Production-ready dataset loader for OCR / document images.
    - Loads from: data/dataset/training_data/images + annotations
    - Validates files
    - Ignores broken images
    - Supports transforms (TorchVision or custom)
    """

    def __init__(self, split="training", transform=None):
        self.transform = transform

        # Resolve dataset root automatically
        dataset_root = PROJECT_ROOT / "data" / "dataset"

        if split == "training":
            split_root = dataset_root / "training_data"
        elif split == "testing":
            split_root = dataset_root / "testing_data"
        else:
            raise ValueError("split must be 'training' or 'testing'")

        self.img_dir = split_root / "images"
        self.ann_dir = split_root / "annotations"

        if not self.img_dir.exists():
            raise RuntimeError(f"Image directory missing: {self.img_dir}")

        # Gather all image paths
        valid_ext = {".jpg", ".jpeg", ".png"}
        self.image_paths = sorted(
            [p for p in self.img_dir.iterdir() if p.suffix.lower() in valid_ext]
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        # Pair each image with annotation
        self.annotation_paths = [
            self.ann_dir / f"{img.stem}.txt" for img in self.image_paths
        ]

        for ann in self.annotation_paths:
            if not ann.exists():
                raise FileNotFoundError(f"Missing annotation: {ann}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        ann_path = self.annotation_paths[idx]

        # Load image (BGR)
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Corrupt image: {img_path}")

        # Convert to RGB (PyTorch convention)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read label text
        with open(ann_path, "r", encoding="utf-8") as f:
            label = f.read().strip()

        # Apply transforms if any
        if self.transform:
            img = self.transform(img)

        return img, label
