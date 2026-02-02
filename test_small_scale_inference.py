"""
Test Inference for Small-Scale Model
Uses folder structure instead of database for reference embeddings.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
import time

sys.path.append(str(Path(__file__).parent))

from backbone import create_model
from dataset import get_transforms
import config


class SmallScaleCardIdentifier:
    """
    Card identifier that works with folder-based dataset structure.
    Designed for models trained with small_scale_train.py.
    """
    
    def __init__(self, model_path: str, data_dir: str, force_rebuild: bool = False):
        """
        Initialize the card identifier.
        
        Args:
            model_path: Path to trained model checkpoint
            data_dir: Path to training data directory with card folders
            force_rebuild: Force rebuild embeddings cache
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        num_classes = checkpoint.get('num_classes', None)
        print(f"Model trained on {num_classes} classes")
        
        self.model = create_model(
            embedding_dim=config.EMBEDDING_DIM,
            gem_p=config.GEM_P_INIT,
            pretrained=False
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = get_transforms(image_size=config.IMAGE_SIZE, augment=False)
        
        # Data directory
        self.data_dir = Path(data_dir)
        
        # Cache path
        cache_dir = Path("model/embeddings_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / "small_scale_reference_embeddings.pkl"
        
        # Build or load reference embeddings
        print("Loading reference embeddings...")
        if self.cache_path.exists() and not force_rebuild:
            self._load_cached_embeddings()
        else:
            if force_rebuild:
                print("Force rebuilding embeddings...")
            self._build_reference_embeddings()
        
        print("Initialization complete!")
    
    def _save_cached_embeddings(self):
        """Save embeddings to cache file."""
        import pickle
        cache_data = {
            'card_ids': self.card_ids,
            'embeddings': self.reference_embeddings,
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved embeddings to cache: {self.cache_path}")
    
    def _load_cached_embeddings(self):
        """Load embeddings from cache file."""
        import pickle
        print(f"Loading embeddings from cache: {self.cache_path}")
        with open(self.cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.card_ids = cache_data['card_ids']
        self.reference_embeddings = cache_data['embeddings']
        print(f"Loaded {len(self.card_ids)} cached embeddings")
    
    def _build_reference_embeddings(self):
        """Build reference embeddings from folder structure."""
        print(f"Building embeddings from {self.data_dir}...")
        
        # Find all card folders
        card_folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
        print(f"Found {len(card_folders)} card folders")
        
        self.card_ids = []
        embeddings_list = []
        
        self.model.eval()
        
        with torch.no_grad():
            for card_folder in tqdm(card_folders, desc="Building embeddings", unit="card"):
                card_id = card_folder.name
                
                # Look for original image first
                img_path = card_folder / f"{card_id}_original.png"
                
                if not img_path.exists():
                    # Fallback to first image in folder
                    images = list(card_folder.glob('*.png')) + list(card_folder.glob('*.jpg'))
                    if not images:
                        continue
                    img_path = images[0]
                
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # Generate embedding
                    embedding = self.model(img_tensor).cpu().numpy()[0]
                    
                    # Store
                    self.card_ids.append(card_id)
                    embeddings_list.append(embedding)
                    
                except Exception as e:
                    print(f"Error processing {card_id}: {e}")
                    continue
        
        # Convert to numpy array
        self.reference_embeddings = np.array(embeddings_list)
        
        print(f"Built {len(self.card_ids)} reference embeddings")
        
        # Save to cache
        self._save_cached_embeddings()
    
    def identify(self, image_path: str, top_k: int = 5):
        """
        Identify a card from an image.
        
        Args:
            image_path: Path to query image
            top_k: Number of top candidates to return
            
        Returns:
            List of candidates with confidence scores
        """
        # Load and process query image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Compute embedding
        with torch.no_grad():
            query_embedding = self.model(img_tensor).cpu().numpy()
        
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        ref_norm = self.reference_embeddings / (np.linalg.norm(self.reference_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity
        similarities = np.dot(ref_norm, query_norm.T).squeeze()
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for i, idx in enumerate(top_indices):
            card_id = self.card_ids[idx]
            
            result = {
                'rank': i + 1,
                'printing_unique_id': card_id,
                'name': card_id,  # Use card_id as name since we don't have database
                'visual_confidence': float(similarities[idx]),
            }
            
            results.append(result)
        
        return results


def test_folder(folder_path: str, model_path: str, data_dir: str, force_rebuild: bool = False):
    """
    Test card identification on all images in a folder.
    
    Args:
        folder_path: Path to folder containing test images
        model_path: Path to trained model
        data_dir: Path to training data directory
        force_rebuild: Force rebuild embeddings cache
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return
    
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
        image_files.extend(folder.glob(ext))
    
    if not image_files:
        print(f"No images found in {folder}")
        return
    
    print(f"Found {len(image_files)} images in {folder}")
    
    # Initialize identifier
    print("\nInitializing card identifier...")
    identifier = SmallScaleCardIdentifier(
        model_path=model_path,
        data_dir=data_dir,
        force_rebuild=force_rebuild
    )
    
    # Test each image
    print("\n" + "=" * 80)
    print("TESTING CARD IDENTIFICATION")
    print("=" * 80)
    
    results = []
    total_inference_time = 0
    
    # Ground truth tracking
    top1_correct = 0
    top5_correct = 0
    total_with_gt = 0
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {img_path.name}")
        print("-" * 80)
        
        # Extract ground truth printing_id from filename (if present)
        filename_parts = img_path.stem.split('.')
        ground_truth_id = filename_parts[0] if len(filename_parts) >= 1 else None
        
        # Check if this looks like a valid printing_id
        has_ground_truth = ground_truth_id and len(ground_truth_id) >= 20 and ground_truth_id.replace('_', '').replace('-', '').isalnum()
        
        try:
            # Time the inference
            start_time = time.time()
            
            # Get top 5 matches
            matches = identifier.identify(str(img_path), top_k=5)
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Validate if we have ground truth
            is_top1_correct = False
            is_top5_correct = False
            
            if has_ground_truth:
                total_with_gt += 1
                
                # Check Top-1
                if matches[0]['printing_unique_id'] == ground_truth_id:
                    is_top1_correct = True
                    is_top5_correct = True
                    top1_correct += 1
                    top5_correct += 1
                else:
                    # Check Top-5
                    for match in matches:
                        if match['printing_unique_id'] == ground_truth_id:
                            is_top5_correct = True
                            top5_correct += 1
                            break
            
            # Display results
            print(f"Top match: {matches[0]['name']}")
            print(f"  Printing ID: {matches[0]['printing_unique_id']}")
            print(f"  Confidence: {matches[0]['visual_confidence']:.4f}")
            print(f"  Inference time: {inference_time*1000:.1f}ms")
            
            # Show top 5
            print(f"\n  Top 5 candidates:")
            for match in matches:
                marker = " ←GT" if has_ground_truth and match['printing_unique_id'] == ground_truth_id else ""
                print(f"    {match['rank']}. {match['name']} ({match['visual_confidence']:.3f}){marker}")
            
            results.append({
                'image': img_path.name,
                'success': True,
                'top_match': matches[0],
                'inference_time_ms': inference_time * 1000,
                'has_ground_truth': has_ground_truth,
                'ground_truth_id': ground_truth_id if has_ground_truth else None,
                'is_top1_correct': is_top1_correct,
                'is_top5_correct': is_top5_correct
            })
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'image': img_path.name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results if r['success'])
    print(f"Successfully identified: {successful}/{len(results)}")
    
    # Accuracy metrics
    if total_with_gt > 0:
        top1_accuracy = (top1_correct / total_with_gt) * 100
        top5_accuracy = (top5_correct / total_with_gt) * 100
        
        print(f"\nAccuracy Metrics (on {total_with_gt} images with ground truth):")
        print(f"  Top-1 Accuracy: {top1_accuracy:.2f}% ({top1_correct}/{total_with_gt})")
        print(f"  Top-5 Accuracy: {top5_accuracy:.2f}% ({top5_correct}/{total_with_gt})")
    
    if successful > 0:
        avg_time = total_inference_time / successful * 1000
        print(f"\nPerformance:")
        print(f"  Average inference time: {avg_time:.1f}ms")
        print(f"  Throughput: {successful/total_inference_time:.1f} images/second")
    
    if successful < len(results):
        print(f"\nFailed images:")
        for r in results:
            if not r['success']:
                print(f"  - {r['image']}: {r['error']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test small-scale card identification")
    parser.add_argument('folder', type=str, help='Path to folder with test images')
    parser.add_argument('--model', type=str, default='model/small_scale_checkpoints/best_model.pth')
    parser.add_argument('--data-dir', type=str, default=r'D:\SIAMESE DATASET\SMALL SCALE OUTPUT',
                        help='Training data directory')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild embeddings cache')
    
    args = parser.parse_args()
    
    test_folder(args.folder, args.model, args.data_dir, args.rebuild)
