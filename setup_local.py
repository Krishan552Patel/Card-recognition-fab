"""
Setup script for local inference.
Downloads required files from Google Drive and sets up the environment.
"""

import os
import sys
from pathlib import Path

def main():
    print("="*60)
    print("Card Recognition Local Setup")
    print("="*60)
    
    # Required paths
    required = {
        'Model': 'checkpoints/best_model.pth',
        'Card JSON': 'fab_cards_data/card-flattened-with-phash.json',
        'Reference Images': 'fab_cards_data/card_images',
    }
    
    print("\n📋 Required files:\n")
    
    missing = []
    for name, path in required.items():
        exists = Path(path).exists()
        status = "✓" if exists else "❌ MISSING"
        print(f"   {status} {name}: {path}")
        if not exists:
            missing.append((name, path))
    
    if missing:
        print("\n" + "="*60)
        print("⚠️  MISSING FILES - Please download from Google Drive:")
        print("="*60)
        print("""
1. Model checkpoint:
   From: Google Drive > CardRecognition_Models > best_model.pth
   To:   checkpoints/best_model.pth

2. Card JSON:
   From: Google Drive > CardData > card-flattened-with-phash.json
   To:   fab_cards_data/card-flattened-with-phash.json

3. Reference images:
   From: Google Drive > CardData > card_images.zip
   To:   fab_cards_data/card_images/ (extract the zip)
""")
        
        # Create directories
        for name, path in missing:
            dir_path = Path(path).parent if not path.endswith('/') else Path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   Created directory: {dir_path}")
        
        print("\nAfter downloading, run: python local_inference.py <image>")
        return False
    
    print("\n✅ All files present!")
    print("\nRun inference with:")
    print("   python local_inference.py path/to/card_image.jpg")
    print("   python local_inference.py --batch path/to/folder/")
    
    # Check Python dependencies
    print("\n📦 Checking dependencies...")
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__}")
    except ImportError:
        print("   ❌ PyTorch not installed: pip install torch torchvision")
    
    try:
        import timm
        print(f"   ✓ timm {timm.__version__}")
    except ImportError:
        print("   ❌ timm not installed: pip install timm")
    
    try:
        import albumentations
        print(f"   ✓ albumentations {albumentations.__version__}")
    except ImportError:
        print("   ❌ albumentations not installed: pip install albumentations")
    
    try:
        from PIL import Image
        print("   ✓ Pillow")
    except ImportError:
        print("   ❌ Pillow not installed: pip install pillow")
    
    print("\n" + "="*60)
    return True


if __name__ == '__main__':
    main()
