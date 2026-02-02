"""
Dataset Loader for FAB Cards
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random


class FABCardDataset(Dataset):
    """
    Dataset for FAB card images organized by printing_unique_id.
    
    Structure:
        D:/SIAMESE DATASET/Images/
            <printing_unique_id_1>/
                rot_000.png
                rot_000_aug_0_hsv_jitter.png
                rot_090.png
                ...
            <printing_unique_id_2>/
                ...
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Root directory containing card folders
            transform: Optional transform to apply to images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get all card folders (each folder = one unique card)
        self.card_folders = [
            d for d in self.root_dir.iterdir()
            if d.is_dir()
        ]
        
        # Create class mapping (folder name -> class index)
        self.class_to_idx = {
            folder.name: idx
            for idx, folder in enumerate(sorted(self.card_folders))
        }
        
        # Build dataset: list of (image_path, class_idx)
        self.samples = []
        for folder in self.card_folders:
            class_idx = self.class_to_idx[folder.name]
            for img_file in folder.glob("*.png"):
                self.samples.append((img_file, class_idx))
        
        print(f"Dataset loaded:")
        print(f"  Unique cards: {len(self.card_folders)}")
        print(f"  Total images: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor
            label: Class index
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_num_classes(self):
        """Return number of unique cards."""
        return len(self.card_folders)


class TripletDataset(Dataset):
    """
    Dataset that returns triplets (anchor, positive, negative).
    Used for triplet loss training.
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Root directory containing card folders
            transform: Optional transform to apply to images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get all card folders
        self.card_folders = [
            d for d in self.root_dir.iterdir()
            if d.is_dir()
        ]
        
        # Build index: class -> list of image paths
        self.class_to_images = {}
        for folder in self.card_folders:
            images = list(folder.glob("*.png"))
            if len(images) > 0:
                self.class_to_images[folder.name] = images
        
        self.classes = list(self.class_to_images.keys())
        
        print(f"Triplet dataset loaded:")
        print(f"  Unique cards: {len(self.classes)}")
    
    def __len__(self):
        return len(self.classes) * 10  # Arbitrary multiplier for epoch length
    
    def __getitem__(self, idx):
        """
        Returns:
            anchor: Image tensor
            positive: Image tensor (same class as anchor)
            negative: Image tensor (different class)
        """
        # Select random class for anchor
        anchor_class = random.choice(self.classes)
        
        # Select anchor and positive from same class
        class_images = self.class_to_images[anchor_class]
        if len(class_images) < 2:
            # If only one image, use it for both anchor and positive
            anchor_path = positive_path = class_images[0]
        else:
            anchor_path, positive_path = random.sample(class_images, 2)
        
        # Select negative from different class
        negative_class = random.choice([c for c in self.classes if c != anchor_class])
        negative_path = random.choice(self.class_to_images[negative_class])
        
        # Load images
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative


def get_transforms(image_size=224, augment=False):
    """
    Get image transforms.
    
    Args:
        image_size: Target image size
        augment: Whether to apply augmentation
        
    Returns:
        transform: torchvision transform
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # NOTE: No horizontal flip - cards have specific orientation!
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(root_dir, batch_size=32, num_workers=4, image_size=224, use_triplet=False):
    """
    Create train and validation dataloaders.
    
    Args:
        root_dir: Root directory containing card folders
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Target image size
        use_triplet: Use triplet dataset instead of classification
        
    Returns:
        train_loader, val_loader
    """
    train_transform = get_transforms(image_size, augment=True)
    val_transform = get_transforms(image_size, augment=False)
    
    if use_triplet:
        train_dataset = TripletDataset(root_dir, transform=train_transform)
        val_dataset = TripletDataset(root_dir, transform=val_transform)
    else:
        # Create TWO separate dataset instances (FIXED - prevents transform conflict)
        train_dataset_full = FABCardDataset(root_dir, transform=train_transform)
        val_dataset_full = FABCardDataset(root_dir, transform=val_transform)
        
        # Create indices for split
        num_samples = len(train_dataset_full.samples)
        indices = list(range(num_samples))
        
        train_size = int(0.8 * num_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Use Subset to maintain separate transforms
        train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    # Create dataloaders
    # CRITICAL FIX: drop_last=True for training to avoid BatchNorm error on single-sample batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches to avoid BatchNorm errors
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # Keep all validation samples
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing FABCardDataset...")
    
    dataset = FABCardDataset(
        root_dir="D:/SIAMESE DATASET/Images",
        transform=get_transforms(224, augment=True)
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Num classes: {dataset.get_num_classes()}")
    
    # Test sample
    img, label = dataset[0]
    print(f"\nSample:")
    print(f"  Image shape: {img.shape}")
    print(f"  Label: {label}")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, val_loader = create_dataloaders(
        root_dir="D:/SIAMESE DATASET/Images",
        batch_size=4,
        num_workers=0
    )
    
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break
