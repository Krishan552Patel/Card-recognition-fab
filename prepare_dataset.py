"""
Prepare Siamese Dataset
Organizes images by unique pHash into folders for training.
Creates augmented versions in each folder.
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

# Configuration
SOURCE_IMAGE_DIR = Path("D:/FABIMAGE")
DATASET_DIR = Path("D:/SIAMESE DATASET/Images")
JSON_PATH = Path("fab_cards_data/card-flattened-with-phash.json")


def load_card_data():
    """Load card data from JSON."""
    print("Loading card data...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    print(f"✓ Loaded {len(cards):,} cards")
    return cards


def group_by_unique_phash(cards):
    """
    Group cards by unique pHash values.
    Only keep one representative per unique pHash.
    """
    print("\nGrouping cards by unique pHash...")
    
    phash_to_card = {}
    cards_with_phash = 0
    
    for card in cards:
        phash = card.get('image_phash')
        if phash and phash.strip():
            cards_with_phash += 1
            # Only keep first card with this pHash
            if phash not in phash_to_card:
                phash_to_card[phash] = card
    
    print(f"✓ Found {cards_with_phash:,} cards with pHash")
    print(f"✓ Found {len(phash_to_card):,} unique pHash values")
    
    return phash_to_card


def create_rotations(image):
    """
    Create 4 rotations of an image: 0°, 90°, 180°, 270°.
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        List of 4 rotated images
    """
    rotations = [
        image,  # 0°
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),  # 90°
        cv2.rotate(image, cv2.ROTATE_180),  # 180°
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 270°
    ]
    return rotations


def apply_augmentation(image, aug_type):
    """
    Apply specific augmentation to image.
    
    Args:
        image: Input image
        aug_type: Type of augmentation
        
    Returns:
        Augmented image
    """
    img = image.copy()
    
    if aug_type == 'hsv_jitter':
        # Convert to HSV and jitter
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] += np.random.uniform(-10, 10)  # Hue
        hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)  # Saturation
        hsv[:, :, 2] *= np.random.uniform(0.8, 1.2)  # Value
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif aug_type == 'blur':
        # Gaussian blur
        kernel_size = np.random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    elif aug_type == 'noise':
        # Gaussian noise
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    elif aug_type == 'brightness':
        # Brightness adjustment
        factor = np.random.uniform(0.7, 1.3)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    
    elif aug_type == 'cutout':
        # Random erasing
        h, w = img.shape[:2]
        erase_h = np.random.randint(h // 10, h // 3)
        erase_w = np.random.randint(w // 10, w // 3)
        y = np.random.randint(0, h - erase_h)
        x = np.random.randint(0, w - erase_w)
        img[y:y+erase_h, x:x+erase_w] = 0
    
    return img


def prepare_dataset(phash_to_card, max_cards=None, augmentations_per_rotation=3):
    """
    Prepare dataset by organizing images into folders.
    
    Args:
        phash_to_card: Dictionary mapping pHash to card data
        max_cards: Maximum number of cards to process (for testing)
        augmentations_per_rotation: Number of augmented versions per rotation
    """
    print(f"\nPreparing dataset in: {DATASET_DIR}")
    DATASET_DIR.mkdir(exist_ok=True, parents=True)
    
    # Limit for testing
    items = list(phash_to_card.items())
    if max_cards:
        items = items[:max_cards]
        print(f"Processing {max_cards} cards for testing...")
    
    stats = {
        'processed': 0,
        'skipped_no_image': 0,
        'total_images_created': 0
    }
    
    augmentation_types = ['hsv_jitter', 'blur', 'noise', 'brightness', 'cutout']
    
    for phash, card in tqdm(items, desc="Processing cards"):
        printing_id = card['printing_unique_id']
        
        # Find source image
        source_image = None
        for ext in ['.png', '.jpg', '.jpeg', '.webp']:
            img_path = SOURCE_IMAGE_DIR / f"{printing_id}{ext}"
            if img_path.exists():
                source_image = img_path
                break
        
        if not source_image:
            stats['skipped_no_image'] += 1
            continue
        
        # Create folder for this card
        card_folder = DATASET_DIR / printing_id
        card_folder.mkdir(exist_ok=True)
        
        # Load image
        img = cv2.imread(str(source_image))
        if img is None:
            stats['skipped_no_image'] += 1
            continue
        
        # Create 4 rotations
        rotations = create_rotations(img)
        
        for rot_idx, rotated_img in enumerate(rotations):
            # Save original rotation
            rot_name = f"rot_{rot_idx * 90:03d}.png"
            cv2.imwrite(str(card_folder / rot_name), rotated_img)
            stats['total_images_created'] += 1
            
            # Create augmented versions
            for aug_idx in range(augmentations_per_rotation):
                # Apply random augmentation
                aug_type = np.random.choice(augmentation_types)
                aug_img = apply_augmentation(rotated_img, aug_type)
                
                aug_name = f"rot_{rot_idx * 90:03d}_aug_{aug_idx}_{aug_type}.png"
                cv2.imwrite(str(card_folder / aug_name), aug_img)
                stats['total_images_created'] += 1
        
        stats['processed'] += 1
    
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"Cards processed:       {stats['processed']:,}")
    print(f"Skipped (no image):    {stats['skipped_no_image']:,}")
    print(f"Total images created:  {stats['total_images_created']:,}")
    print(f"Dataset location:      {DATASET_DIR}")
    print("=" * 60)
    
    return stats


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Siamese dataset")
    parser.add_argument(
        '--max-cards',
        type=int,
        help='Maximum number of cards to process (for testing)'
    )
    parser.add_argument(
        '--augmentations',
        type=int,
        default=3,
        help='Number of augmentations per rotation (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Load and group cards
    cards = load_card_data()
    phash_to_card = group_by_unique_phash(cards)
    
    # Prepare dataset
    prepare_dataset(
        phash_to_card,
        max_cards=args.max_cards,
        augmentations_per_rotation=args.augmentations
    )


if __name__ == "__main__":
    main()
