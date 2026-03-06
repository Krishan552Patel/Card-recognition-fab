"""
Augmentation pipelines for controlled card sorting environment.
Source of truth: Card_Recognition_Training.ipynb (V6), cell 10.
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_controlled_augmentation(size: int = 224) -> A.Compose:
    """
    Augmentation for a controlled card-sorting environment where the card is
    always centred in frame. Covers:
      - Slight geometric variation (±5° rotation, ±5% scale)
      - Camera blur (motion / gaussian)
      - Lighting variation (brightness/contrast, gamma, CLAHE)
      - White-balance / colour shift (HSV, RGB shift, ColorJitter)
      - Sensor noise (Gaussian, ISO)
    """
    return A.Compose([
        A.Resize(size, size),

        # Minimal geometric — card placement variation only
        A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),
        A.Affine(scale=(0.95, 1.05), p=0.2),

        # Blur — camera focus and slight motion
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.4),

        # Lighting — ambient light conditions
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.CLAHE(clip_limit=3.0, p=1.0),
        ], p=0.6),

        # Colour — LED vs daylight, white balance
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=1.0),
        ], p=0.5),

        # Noise — camera sensor noise
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.ISONoise(intensity=(0.1, 0.3), p=1.0),
        ], p=0.3),

        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(size: int = 224) -> A.Compose:
    """Clean validation / inference transforms — no augmentation."""
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])
