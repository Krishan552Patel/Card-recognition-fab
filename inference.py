"""
Inference Pipeline for Card Identification
Combines visual matching with database verification for high confidence.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backbone import create_model
from dataset import get_transforms
import config
from fab_cards_db import DatabaseManager


class CardIdentifier:
    """
    Two-stage card identification system:
    1. Visual matching: Find top-k candidates using embedding similarity
    2. Database verification: Use metadata to confirm and rank candidates
    """
    
    def __init__(self, model_path: str, db_path: str = "fab_cards_data/fab_cards.db", force_rebuild: bool = False):
        """
        Initialize the card identifier.
        
        Args:
            model_path: Path to trained model checkpoint
            db_path: Path to card database
            force_rebuild: Force rebuild embeddings even if cache exists
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = create_model(
            embedding_dim=config.EMBEDDING_DIM,
            gem_p=config.GEM_P_INIT,
            pretrained=False
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load database
        print(f"Loading database from {db_path}...")
        self.db = DatabaseManager(db_path)
        
        # Image transforms
        self.transform = get_transforms(image_size=config.IMAGE_SIZE, augment=False)
        
        # Image directory - FIXED to use correct training data path
        self.image_dir = Path(config.IMAGE_DIR)
        
        # Cache path
        self.cache_dir = Path("model/embeddings_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "reference_embeddings.pkl"
        
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
            'card_metadata': self.card_metadata
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
        self.card_metadata = cache_data['card_metadata']
        print(f"Loaded {len(self.card_ids)} cached embeddings")
    
    def _build_reference_embeddings(self):
        """Build reference embeddings for all cards in database with progress tracking."""
        from tqdm import tqdm
        
        # Get all cards with image URLs
        all_cards = self.db.get_all_cards()
        cards_with_images = [c for c in all_cards if c.get('image_url')]
        
        print(f"Found {len(cards_with_images)} cards with images")
        
        self.card_metadata = {}
        self.card_ids = []
        embeddings_list = []
        
        self.model.eval()
        
        # Add progress bar with ASCII characters only
        with torch.no_grad():
            for card in tqdm(cards_with_images, desc="Building embeddings", unit="card", ascii=True):
                printing_id = card['printing_unique_id']
                
                # Get image path from dataset directory
                img_folder = self.image_dir / printing_id
                
                # Look for the base rotation image (rot_000.png)
                img_path = img_folder / 'rot_000.png'
                
                if not img_path.exists():
                    # Fallback to first image in folder
                    images = list(img_folder.glob('*.png'))
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
                    self.card_ids.append(printing_id)
                    embeddings_list.append(embedding)
                    
                    # Store metadata
                    self.card_metadata[printing_id] = {
                        'name': card.get('name'),
                        'set_id': card.get('set_id'),
                        'edition': card.get('edition'),
                        'foiling': card.get('foiling'),
                        'rarity': card.get('rarity'),
                        'image_url': card.get('image_url'),
                        'card_id': card.get('card_id'),
                        'pitch': card.get('pitch'),
                        'color': card.get('color'),
                        'cost': card.get('cost'),
                        'power': card.get('power'),
                        'defense': card.get('defense'),
                        'types': card.get('types')
                    }
                    
                except Exception as e:
                    # Skip cards with errors
                    continue
        
        # Convert to numpy array for efficient similarity search
        self.reference_embeddings = np.array(embeddings_list)
        
        print(f"Built {len(self.card_ids)} reference embeddings")
        
        # Save to cache
        self._save_cached_embeddings()
    
    def identify(
        self,
        image_path: str,
        top_k: int = 5,
        return_details: bool = True
    ) -> List[Dict]:
        """
        Identify a card from an image.
        
        Args:
            image_path: Path to query image
            top_k: Number of top candidates to return
            return_details: Include detailed metadata
            
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
            metadata = self.card_metadata[card_id]
            
            result = {
                'rank': i + 1,
                'printing_unique_id': card_id,
                'visual_confidence': float(similarities[idx]),
                'name': metadata['name'],
            }
            
            if return_details:
                result.update({
                    'card_id': metadata.get('card_id'),
                    'set_id': metadata.get('set_id'),
                    'edition': metadata.get('edition'),
                    'foiling': metadata.get('foiling'),
                    'rarity': metadata.get('rarity'),
                    'pitch': metadata.get('pitch'),
                    'color': metadata.get('color'),
                    'cost': metadata.get('cost'),
                    'power': metadata.get('power'),
                    'defense': metadata.get('defense'),
                    'types': metadata.get('types'),
                    'image_url': metadata.get('image_url')
                })
            
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Demo
    import argparse
    
    parser = argparse.ArgumentParser(description="Card Identification Demo")
    parser.add_argument('image', type=str, help='Path to card image')
    parser.add_argument('--model', type=str, default='model/checkpoints/best_model.pth')
    parser.add_argument('--top-k', type=int, default=5, help='Number of candidates')
    
    args = parser.parse_args()
    
    identifier = CardIdentifier(model_path=args.model)
    
    print("\n" + "=" * 80)
    print("CARD IDENTIFICATION")
    print("=" * 80)
    
    results = identifier.identify(args.image, top_k=args.top_k)
    
    print(f"\nTop {args.top_k} matches:")
    for result in results:
        print(f"\n{result['rank']}. {result['name']}")
        print(f"   ID: {result['printing_unique_id']}")
        print(f"   Set: {result['set_id']} | Edition: {result['edition']}")
        print(f"   Confidence: {result['visual_confidence']:.4f}")
