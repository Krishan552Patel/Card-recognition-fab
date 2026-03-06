"""
CardDataset — flat image folder with on-the-fly 0/90/180/270 degree rotation.
Source of truth: Card_Recognition_Training.ipynb (V6), cell 15.
"""

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_ROTATIONS = [0, 90, 180, 270]


class CardDataset(Dataset):
    """
    Loads card images from a flat directory (one image per card identity).

    Each image is exposed as four samples — one per 90-degree rotation — so the
    model learns to recognise cards in any orientation without pre-generating
    augmented copies on disk.

    Args:
        image_dir: Path to folder containing card images (flat, no subfolders).
        transform:  Albumentations transform applied after rotation.
        rotations:  List of rotations in degrees to include (default all four).
    """

    def __init__(self, image_dir, transform, rotations=None):
        if rotations is None:
            rotations = _ROTATIONS
        self.images = sorted(
            [p for p in Path(image_dir).iterdir() if p.suffix.lower() in _IMAGE_EXTS]
        )
        self.samples = [
            (img_idx, rot)
            for img_idx in range(len(self.images))
            for rot in rotations
        ]
        self.transform = transform
        print(
            f"CardDataset: {len(self.images)} cards x {len(rotations)} rotations "
            f"= {len(self.samples)} samples"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, rot = self.samples[idx]
        try:
            img = np.array(Image.open(self.images[img_idx]).convert("RGB"))
            if rot == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif rot == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return self.transform(image=img)["image"], img_idx
        except Exception:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

    def get_num_classes(self):
        return len(self.images)
