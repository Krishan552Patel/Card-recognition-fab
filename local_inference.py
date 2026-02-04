"""
Local Card Recognition Inference (CPU Only)
============================================
Test the trained model on your PC without using GPU.

Usage:
    python local_inference.py path/to/image.jpg
    python local_inference.py --batch path/to/folder/

Requirements:
    pip install torch torchvision timm albumentations pillow numpy
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for optional dependencies
try:
    import timm
except ImportError:
    print("Please install timm: pip install timm")
    sys.exit(1)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Please install albumentations: pip install albumentations")
    sys.exit(1)


# ============================================
# Model Architecture (must match training)
# ============================================

class GeM(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
    
    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=1e-6).pow(self.p), 1
        ).pow(1./self.p).view(x.size(0), -1)


class CardNet(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            'mobilenetv3_large_100', 
            pretrained=False, 
            num_classes=0, 
            global_pool=''
        )
        
        with torch.no_grad():
            self.n_feat = self.backbone(torch.randn(1, 3, 224, 224)).shape[1]
        
        self.gem = GeM()
        self.head = nn.Sequential(
            nn.Linear(self.n_feat, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )
    
    def forward(self, x):
        features = self.gem(self.backbone(x))
        emb = self.head(features)
        return F.normalize(emb, p=2, dim=1)


# ============================================
# Transforms
# ============================================

def get_transforms(size=224):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ============================================
# Card Identifier
# ============================================

class CardIdentifier:
    def __init__(self, model_path, card_json_path, image_dir):
        print("Loading model...")
        self.device = torch.device('cpu')
        self.transform = get_transforms()
        
        # Load model
        self.model = CardNet(emb_dim=512)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ Model loaded (epoch {checkpoint.get('epoch', '?')})")
        
        # Load card lookup
        print("Loading card database...")
        with open(card_json_path, 'r', encoding='utf-8') as f:
            all_cards = json.load(f)
        self.card_lookup = {c['printing_unique_id']: c for c in all_cards}
        print(f"✓ {len(all_cards):,} cards loaded")
        
        # Build reference embeddings
        print("Building reference embeddings (this may take a few minutes on CPU)...")
        self.ref_embeddings, self.ref_names = self._build_references(image_dir)
        print(f"✓ {len(self.ref_names):,} reference embeddings built")
    
    def _build_references(self, image_dir):
        image_dir = Path(image_dir)
        images = sorted([f for f in image_dir.iterdir() 
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']])
        
        embeddings, names = [], []
        with torch.no_grad():
            for i, img_path in enumerate(images):
                if i % 500 == 0:
                    print(f"   Processing {i}/{len(images)}...")
                try:
                    img = np.array(Image.open(img_path).convert('RGB'))
                    tensor = self.transform(image=img)['image'].unsqueeze(0)
                    emb = self.model(tensor)
                    embeddings.append(emb)
                    names.append(img_path.stem)
                except Exception as e:
                    print(f"   Warning: Failed to process {img_path.name}: {e}")
        
        return torch.cat(embeddings, dim=0), names
    
    def get_card_name(self, printing_id):
        card = self.card_lookup.get(printing_id, {})
        name = card.get('name', printing_id[:20])
        foil = card.get('foiling', '')
        if foil and foil != 'S':
            return f"{name} ({foil})"
        return name
    
    def identify(self, image_path, top_k=5):
        """Identify a card from an image file."""
        img = np.array(Image.open(image_path).convert('RGB'))
        tensor = self.transform(image=img)['image'].unsqueeze(0)
        
        with torch.no_grad():
            query_emb = self.model(tensor)
        
        sims = F.cosine_similarity(query_emb, self.ref_embeddings)
        top_indices = sims.argsort(descending=True)[:top_k]
        
        results = []
        for idx in top_indices:
            printing_id = self.ref_names[idx]
            card = self.card_lookup.get(printing_id, {})
            results.append({
                'name': card.get('name', 'Unknown'),
                'printing_id': printing_id,
                'confidence': sims[idx].item(),
                'set': card.get('set_id', ''),
                'foil': card.get('foiling', ''),
            })
        
        return results


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Local Card Recognition')
    parser.add_argument('image', nargs='?', help='Path to image file')
    parser.add_argument('--batch', help='Path to folder of images')
    parser.add_argument('--model', default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--json', default='fab_cards_data/card-flattened-with-phash.json',
                       help='Path to card JSON')
    parser.add_argument('--images', default='fab_cards_data/card_images',
                       help='Path to reference images folder')
    parser.add_argument('--top', type=int, default=5, help='Number of matches to show')
    
    args = parser.parse_args()
    
    if not args.image and not args.batch:
        parser.print_help()
        print("\nExample:")
        print("  python local_inference.py my_card.jpg")
        print("  python local_inference.py --batch test_images/")
        return
    
    # Check paths
    model_path = Path(args.model)
    json_path = Path(args.json)
    images_path = Path(args.images)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Download from Google Drive: CardRecognition_Models/best_model.pth")
        return
    
    if not json_path.exists():
        print(f"❌ Card JSON not found: {json_path}")
        return
    
    if not images_path.exists():
        print(f"❌ Reference images not found: {images_path}")
        return
    
    # Initialize identifier
    identifier = CardIdentifier(model_path, json_path, images_path)
    
    # Process images
    if args.image:
        images = [Path(args.image)]
    else:
        batch_path = Path(args.batch)
        images = [f for f in batch_path.iterdir() 
                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    print("\n" + "="*60)
    
    for img_path in images:
        if not img_path.exists():
            print(f"❌ Image not found: {img_path}")
            continue
        
        print(f"\n🎴 {img_path.name}")
        print("-"*40)
        
        results = identifier.identify(img_path, top_k=args.top)
        
        for i, r in enumerate(results):
            status = "✓" if i == 0 else " "
            print(f"   {status} {i+1}. {r['name']} ({r['confidence']*100:.1f}%)")
            print(f"        Set: {r['set']} | Foil: {r['foil']}")
    
    print("\n" + "="*60)
    print("Done!")


if __name__ == '__main__':
    main()
