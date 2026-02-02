"""
Enhanced Multi-Modal Inference for Small-Scale Model
Combines: Visual Embedding + Perceptual Hash + OCR + Fuzzy Matching
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import imagehash
from rapidfuzz import fuzz
import json

sys.path.append(str(Path(__file__).parent))

from backbone import create_model
from dataset import get_transforms
import config

# Optional: Try to import OCR
try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
    # Initialize PaddleOCR (we'll create instance in __init__)
except ImportError:
    OCR_AVAILABLE = False
    PaddleOCR = None
    print("  PaddleOCR not available, OCR will be disabled")


class EnhancedCardIdentifier:
    """
    Multi-modal card identifier with:
    1. Visual embedding similarity
    2. Perceptual hash matching
    3. OCR + fuzzy text matching
    """
    
    def __init__(self, model_path: str, data_dir: str, json_db_path: str = None, force_rebuild: bool = False, disable_ocr: bool = False):
        """
        Initialize the enhanced identifier.
        
        Args:
            model_path: Path to trained model checkpoint
            data_dir: Path to training data directory with card folders
            json_db_path: Path to JSON database with card metadata and phash
            force_rebuild: Force rebuild embeddings cache
            disable_ocr: Skip PaddleOCR initialization for faster startup
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load JSON database if provided
        self.card_db = {}
        self.full_db_index = []  # Pre-computed index for fast searching
        self.name_to_cards = {}  # Fast lookup: card name -> list of card entries
        if json_db_path and Path(json_db_path).exists():
            print(f"Loading card database from {json_db_path}...")
            with open(json_db_path, 'r', encoding='utf-8') as f:
                cards_list = json.load(f)
                # Index by printing_unique_id
                for card in cards_list:
                    pid = card.get('printing_unique_id')
                    if pid:
                        self.card_db[pid] = card
            print(f"Loaded {len(self.card_db)} cards from database")
            
            # Pre-compute searchable text index for FAST fuzzy matching
            print("Building full database search index...")
            for card_id, card_data in self.card_db.items():
                name = card_data.get('name', '')
                functional_text = card_data.get('functional_text_plain', '')
                artists = ' '.join(card_data.get('artists', []))
                type_text = card_data.get('type_text', '')
                
                # Store lowercase searchable text for fast matching
                searchable = f"{name} {functional_text} {artists} {type_text}".lower()
                entry = {
                    'id': card_id,
                    'name': name,
                    'name_lower': name.lower(),
                    'searchable': searchable,
                    'data': card_data
                }
                self.full_db_index.append(entry)
                
                # Build name lookup for instant exact matching
                name_key = name.lower()
                if name_key not in self.name_to_cards:
                    self.name_to_cards[name_key] = []
                self.name_to_cards[name_key].append(entry)
            print(f"Built search index for {len(self.full_db_index)} cards ({len(self.name_to_cards)} unique names)")
        
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
        self.cache_path = cache_dir / "enhanced_reference_data.pkl"
        
        # Initialize PaddleOCR (skip if disabled for speed)
        self.ocr_reader = None
        if OCR_AVAILABLE and not disable_ocr:
            print("Initializing PaddleOCR...")
            self.ocr_reader = PaddleOCR(use_textline_orientation=True, lang='en')
        elif disable_ocr:
            print("OCR disabled for faster inference")
        
        # Build or load reference data
        print("Loading reference data...")
        if self.cache_path.exists() and not force_rebuild:
            self._load_cached_data()
        else:
            if force_rebuild:
                print("Force rebuilding reference data...")
            self._build_reference_data()
        
        print("Initialization complete!")
    
    def _save_cached_data(self):
        """Save reference data to cache."""
        import pickle
        cache_data = {
            'card_ids': self.card_ids,
            'embeddings': self.reference_embeddings,
            'phashes': self.reference_phashes,
            'card_names': self.card_names,
            'searchable': self.reference_searchable
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved reference data to cache: {self.cache_path}")
    
    def _load_cached_data(self):
        """Load reference data from cache."""
        import pickle
        print(f"Loading from cache: {self.cache_path}")
        with open(self.cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.card_ids = cache_data['card_ids']
        self.reference_embeddings = cache_data['embeddings']
        self.reference_phashes = cache_data['phashes']
        self.card_names = cache_data['card_names']
        self.reference_searchable = cache_data.get('searchable', [])
        
        # Rebuild searchable if missing from old cache
        if not self.reference_searchable:
            print("Building searchable text index...")
            self.reference_searchable = []
            for card_id in self.card_ids:
                metadata = self.card_db.get(card_id, {})
                card_name = self.card_names.get(card_id, card_id)
                searchable = f"{card_name} {metadata.get('functional_text_plain', '')} {' '.join(metadata.get('artists', []))}"
                self.reference_searchable.append(searchable.lower())
        
        print(f"Loaded {len(self.card_ids)} cached entries")
    
    def _build_reference_data(self):
        """Build reference embeddings and phashes from folder structure."""
        print(f"Building reference data from {self.data_dir}...")
        
        # Find all card folders
        card_folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
        print(f"Found {len(card_folders)} card folders")
        
        self.card_ids = []
        embeddings_list = []
        self.reference_phashes = []
        self.card_names = {}
        self.reference_searchable = []
        
        self.model.eval()
        
        with torch.no_grad():
            for card_folder in tqdm(card_folders, desc="Building reference data", unit="card"):
                card_id = card_folder.name
                
                # Look for original image first
                img_path = card_folder / f"{card_id}_original.png"
                
                if not img_path.exists():
                    # Fallback to first image
                    images = list(card_folder.glob('*.png')) + list(card_folder.glob('*.jpg'))
                    if not images:
                        continue
                    img_path = images[0]
                
                try:
                    # Load image
                    img = Image.open(img_path).convert('RGB')
                    
                    # Generate embedding
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    embedding = self.model(img_tensor).cpu().numpy()[0]
                    
                    # Compute perceptual hash
                    phash = imagehash.phash(img, hash_size=16)  # 256-bit hash
                    
                    # Get card metadata from database
                    metadata = self.card_db.get(card_id, {})
                    card_name = metadata.get('name', card_id)
                    
                    # Build searchable text for OCR fuzzy matching
                    searchable = f"{card_name} {metadata.get('functional_text_plain', '')} {' '.join(metadata.get('artists', []))}"
                    
                    # Store
                    self.card_ids.append(card_id)
                    embeddings_list.append(embedding)
                    self.reference_phashes.append(str(phash))
                    self.card_names[card_id] = card_name
                    self.reference_searchable.append(searchable.lower())
                    
                except Exception as e:
                    print(f"Error processing {card_id}: {e}")
                    continue
        
        # Convert to numpy array
        self.reference_embeddings = np.array(embeddings_list)
        
        print(f"Built {len(self.card_ids)} reference entries")
        
        # Save to cache
        self._save_cached_data()
    
    def _compute_phash_similarity(self, hash1_str: str, hash2_str: str) -> float:
        """
        Compute similarity between two phashes.
        Returns 0.0 to 1.0 (1.0 = identical)
        """
        try:
            hash1 = imagehash.hex_to_hash(hash1_str)
            hash2 = imagehash.hex_to_hash(hash2_str)
            
            # Hamming distance (lower = more similar)
            distance = hash1 - hash2
            max_distance = len(hash1.hash.flatten())
            
            # Convert to similarity score
            similarity = 1.0 - (distance / max_distance)
            return similarity
        except:
            return 0.0
    
    def _extract_card_text_ocr(self, image: Image.Image, save_debug_path: str = None) -> tuple[str, str]:
        """
        Extract ALL text from card image using PaddleOCR.
        
        Args:
            image: PIL Image to process
            save_debug_path: If provided, saves debug image with OCR boxes to this path
            
        Returns: (all_text_combined, debug_info)
        """
        if not OCR_AVAILABLE or self.ocr_reader is None:
            return "", " PaddleOCR not installed (pip install paddleocr)"
        
        try:
            # Convert PIL to numpy array for PaddleOCR
            import numpy as np
            img_array = np.array(image)
            
            # Run PaddleOCR on the ENTIRE card image
            result = self.ocr_reader.ocr(img_array)
            
            # Debug: print raw result type and structure
            print(f"   OCR Raw Result Type: {type(result)}")
            if result:
                print(f"  🔍 OCR Result Length: {len(result)}")
                if result[0]:
                    print(f"   OCR Detections: {len(result[0])} text regions found")
                else:
                    print(f"   OCR result[0] is: {result[0]}")
            else:
                print(f"   OCR result is: {result}")
            
            # Save debug visualization if requested
            if save_debug_path:
                self._save_ocr_debug_image(image, result, save_debug_path)
            
            if not result or not result[0]:
                return "", " No text detected in card image"
            
            ocr_result = result[0]
            
            # New PaddleOCR format: OCRResult is dict-like with rec_texts and rec_scores
            if hasattr(ocr_result, 'get') or hasattr(ocr_result, 'keys'):
                # Dict-like OCRResult object
                rec_texts = ocr_result.get('rec_texts', []) if hasattr(ocr_result, 'get') else getattr(ocr_result, 'rec_texts', [])
                rec_scores = ocr_result.get('rec_scores', []) if hasattr(ocr_result, 'get') else getattr(ocr_result, 'rec_scores', [])
                
                if rec_texts:
                    detected_texts = [str(t) for t in rec_texts if t and str(t).strip()]
                    scores = [float(s) for s in rec_scores] if rec_scores else [1.0] * len(detected_texts)
                    
                    print(f"  🔍 OCR detected {len(detected_texts)} texts")
                    for i, (text, score) in enumerate(zip(detected_texts[:5], scores[:5])):
                        print(f"      [{i}] '{text}' (conf: {score:.2f})")
                    
                    if detected_texts:
                        all_text = " ".join(detected_texts)
                        avg_confidence = sum(scores) / len(scores) if scores else 1.0
                        debug_info = f"✓ Extracted {len(detected_texts)} text regions (avg conf: {avg_confidence:.2f})"
                        return all_text.strip(), debug_info
            
            # Fallback: try old list format
            detected_texts = []
            total_confidence = 0
            
            for idx, line in enumerate(ocr_result):
                try:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        text_data = line[1]
                        if isinstance(text_data, (list, tuple)) and len(text_data) >= 2:
                            text = str(text_data[0])
                            confidence = float(text_data[1])
                        elif isinstance(text_data, str):
                            text = text_data
                            confidence = 1.0
                        else:
                            continue
                        
                        if text and str(text).strip():
                            detected_texts.append(str(text))
                            total_confidence += confidence
                except:
                    continue
            
            if not detected_texts:
                return "", " No valid text detected in card image"
            
            all_text = " ".join(detected_texts)
            avg_confidence = total_confidence / len(detected_texts) if detected_texts else 0
            
            debug_info = f"✓ Extracted {len(detected_texts)} text regions (avg conf: {avg_confidence:.2f})"
            
            return all_text.strip(), debug_info
            
        except Exception as e:
            return "", f" PaddleOCR Error: {str(e)}"
    
    def _save_ocr_debug_image(self, image: Image.Image, ocr_result, save_path: str):
        """Save a debug image showing OCR bounding boxes and detected text."""
        import cv2
        import numpy as np
        
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Draw OCR results
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                if line and len(line) >= 2:
                    try:
                        # Get bounding box points
                        box = line[0]
                        if box and len(box) >= 4:
                            pts = np.array(box, dtype=np.int32)
                            
                            # Draw the bounding box
                            cv2.polylines(img_cv, [pts], True, (0, 255, 0), 2)
                            
                            # Get text and confidence
                            text_info = line[1] if isinstance(line[1], (list, tuple)) else line
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = str(text_info[0])
                                confidence = float(text_info[1])
                                
                                # Draw text label above box
                                x, y = int(pts[0][0]), int(pts[0][1]) - 10
                                label = f"{text[:30]} ({confidence:.2f})"
                                cv2.putText(img_cv, label, (x, y), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    except Exception:
                        continue
        else:
            # No OCR results - add watermark
            h, w = img_cv.shape[:2]
            cv2.putText(img_cv, "NO TEXT DETECTED", (w//4, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Save debug image
        cv2.imwrite(save_path, img_cv)
        print(f"  📸 Saved OCR debug image: {save_path}")
    
    def _fuzzy_match_card_text(self, ocr_text: str, card_idx: int) -> float:
        """
        Fuzzy match OCR text against pre-built searchable text.
        Uses token_set_ratio for better handling of word reordering.
        Returns similarity score (0.0 to 1.0).
        """
        if not ocr_text or card_idx >= len(self.reference_searchable):
            return 0.0
        
        searchable = self.reference_searchable[card_idx]
        if not searchable:
            return 0.0
        
        # Use token_set_ratio - handles word reordering and partial matches well
        score = fuzz.token_set_ratio(ocr_text.lower(), searchable) / 100.0
        return score
    
    def _search_full_db_by_text(self, ocr_text: str, top_k: int = 10) -> list:
        """
        SMART search: Tries exact name match first (instant O(1)), then fuzzy if needed.
        Early exits when high-confidence matches are found.
        
        Returns: List of (card_id, card_name, score, metadata) tuples, sorted by score desc
        """
        if not ocr_text or not self.full_db_index:
            return []
        
        from rapidfuzz import fuzz as rfuzz
        
        ocr_lower = ocr_text.lower()
        matches = []
        
        # TIER 1: INSTANT - Check if any card name appears exactly in the OCR text
        # This is O(n) on unique names but with simple string matching - very fast
        for card_name, entries in self.name_to_cards.items():
            if card_name in ocr_lower and len(card_name) >= 3:  # Min 3 chars to avoid false positives
                # Exact match found! Add all printings of this card
                for entry in entries:
                    matches.append((entry['id'], entry['name'], 0.95, entry['data']))
        
        # If we found exact matches, return immediately - no need for fuzzy search
        if matches:
            matches.sort(key=lambda x: x[2], reverse=True)
            return matches[:top_k]
        
        # TIER 2: FAST FUZZY - Only if no exact matches found
        # Use RapidFuzz to find similar card names
        from rapidfuzz import process
        
        # Build name list only once (could cache this)
        name_choices = list(self.name_to_cards.keys())
        
        # Find cards with similar names to the OCR text
        name_matches = process.extract(
            ocr_lower, 
            name_choices, 
            scorer=rfuzz.partial_ratio,
            limit=20  # Only check top 20 name candidates
        )
        
        for card_name, score, _ in name_matches:
            if score >= 50:  # Only consider if at least 50% match
                for entry in self.name_to_cards[card_name]:
                    # Quick full-text check for better accuracy
                    text_score = rfuzz.token_set_ratio(ocr_lower, entry['searchable'])
                    final_score = max(score, text_score) / 100.0
                    
                    if final_score > 0.3:
                        matches.append((entry['id'], entry['name'], final_score, entry['data']))
                        
                # Early exit if we found high-confidence matches
                if matches and max(m[2] for m in matches) > 0.7:
                    break
        
        # Sort by score descending
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:top_k]
    
    def identify(
        self,
        image_input,  # Can be file path (str) or PIL Image
        top_k: int = 5,
        visual_weight: float = 0.5,  # 50% visual
        phash_weight: float = 0.5,   # 50% pHash
        ocr_weight: float = 0.3,     # OCR weight when used
        ocr_threshold: float = 0.50, # Only run OCR if below this confidence
        ocr_debug_path: str = None
    ):
        """
        Identify a card using multi-modal approach.
        
        OCR is only used when visual+pHash confidence is below ocr_threshold.
        Fuzzy matching is limited to top 10 visual candidates for speed.
        
        Args:
            image_input: Path to query image (str) or PIL Image object
            top_k: Number of top candidates to return
            visual_weight: Weight for visual embedding similarity (0.0-1.0)
            phash_weight: Weight for perceptual hash similarity (0.0-1.0)
            ocr_weight: Weight for OCR fuzzy matching when used (0.0-1.0)
            ocr_threshold: Only run OCR if confidence below this value (default 0.50)
            
        Returns:
            List of candidates with multi-modal confidence scores
        """
        # Load and process query image - accept both path and PIL Image
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            raise ValueError(f"image_input must be str (path) or PIL.Image, got {type(image_input)}")
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 1. VISUAL EMBEDDING SIMILARITY
        with torch.no_grad():
            query_embedding = self.model(img_tensor).cpu().numpy()
        
        # Normalize for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        ref_norm = self.reference_embeddings / (np.linalg.norm(self.reference_embeddings, axis=1, keepdims=True) + 1e-8)
        
        visual_similarities = np.dot(ref_norm, query_norm.T).squeeze()
        
        # 2. PERCEPTUAL HASH SIMILARITY
        query_phash = str(imagehash.phash(img, hash_size=16))
        phash_similarities = np.array([
            self._compute_phash_similarity(query_phash, ref_phash)
            for ref_phash in self.reference_phashes
        ])
        
        # Initial combined score (visual + pHash only)
        initial_combined = (visual_weight * visual_similarities + phash_weight * phash_similarities) / (visual_weight + phash_weight)
        
        # Get initial top score to decide if OCR is needed
        top_initial_score = np.max(initial_combined)
        
        # 3. OCR + FUZZY MATCHING - ONLY if confidence is LOW
        ocr_text = ""
        ocr_debug = ""
        ocr_similarities = np.zeros(len(self.card_ids))
        ocr_match_details = []
        full_db_matches = []
        ocr_used = False
        
        if top_initial_score < ocr_threshold and self.ocr_reader is not None:
            ocr_used = True
            print(f"  [OCR] Low confidence ({top_initial_score:.3f} < {ocr_threshold}), running OCR...")
            
            # Extract text via OCR
            ocr_text, ocr_debug = self._extract_card_text_ocr(img, save_debug_path=ocr_debug_path)
            
            if ocr_text:
                # Only fuzzy match against TOP 50 visual candidates (not all 14k cards!)
                top_50_indices = np.argsort(initial_combined)[::-1][:50]
                
                for idx in top_50_indices:
                    similarity = self._fuzzy_match_card_text(ocr_text, idx)
                    ocr_similarities[idx] = similarity
                    ocr_match_details.append({
                        'card_id': self.card_ids[idx],
                        'card_name': self.card_names.get(self.card_ids[idx], self.card_ids[idx]),
                        'ocr_score': similarity
                    })
                
                # Sort by OCR score
                ocr_match_details.sort(key=lambda x: x['ocr_score'], reverse=True)
                
                print(f"  [OCR] Extracted: \"{ocr_text[:50]}...\"")
                print(f"  [OCR] Top match: {ocr_match_details[0]['card_name']} ({ocr_match_details[0]['ocr_score']:.3f})")
        
        # COMBINE SCORES
        if ocr_used and ocr_text:
            # Use all three modalities
            total_weight = visual_weight + phash_weight + ocr_weight
            combined_scores = (
                (visual_weight / total_weight) * visual_similarities +
                (phash_weight / total_weight) * phash_similarities +
                (ocr_weight / total_weight) * ocr_similarities
            )
        else:
            # Just visual + pHash
            combined_scores = initial_combined
        
        # Get top-k
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        # Build results
        results = []
        for i, idx in enumerate(top_indices):
            card_id = self.card_ids[idx]
            card_name = self.card_names.get(card_id, card_id)
            
            # Get database metadata if available
            metadata = self.card_db.get(card_id, {})
            
            result = {
                'rank': i + 1,
                'printing_unique_id': card_id,
                'card_id': card_id,  # Alias for compatibility
                'name': card_name,
                'combined_score': float(combined_scores[idx]),
                'combined_confidence': float(combined_scores[idx]),
                'visual_similarity': float(visual_similarities[idx]),
                'visual_confidence': float(visual_similarities[idx]),
                'phash_similarity': float(phash_similarities[idx]),
                'phash_confidence': float(phash_similarities[idx]),
                'ocr_confidence': float(ocr_similarities[idx]),
                'ocr_used': ocr_used,
                'ocr_text': ocr_text if i == 0 else None,  # Only show for top match
                'ocr_debug': ocr_debug if i == 0 else None,  # Debug info
                'ocr_top_matches': ocr_match_details[:5] if i == 0 and ocr_match_details else None,
                'full_db_ocr_matches': full_db_matches[:5] if i == 0 and full_db_matches else None,
                # Add metadata if available
                'set_id': metadata.get('set_id'),
                'edition': metadata.get('edition'),
                'foiling': metadata.get('foiling'),
                'rarity': metadata.get('rarity'),
            }
            
            results.append(result)
        
        return results


def test_folder(folder_path: str, model_path: str, data_dir: str, json_db_path: str = None, force_rebuild: bool = False, debug_ocr: bool = False):
    """Test enhanced card identification on all images in a folder."""
    folder = Path(folder_path)
    
    # Create debug output folder if needed
    debug_dir = None
    if debug_ocr:
        debug_dir = folder / "ocr_debug"
        debug_dir.mkdir(exist_ok=True)
        print(f"\nOCR Debug Mode: Saving debug images to {debug_dir}")
    
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
    print("\nInitializing enhanced card identifier...")
    identifier = EnhancedCardIdentifier(
        model_path=model_path,
        data_dir=data_dir,
        json_db_path=json_db_path,
        force_rebuild=force_rebuild
    )
    
    # Test each image
    print("\n" + "=" * 80)
    print("ENHANCED MULTI-MODAL CARD IDENTIFICATION")
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
        
        # Extract ground truth
        filename_parts = img_path.stem.split('.')
        ground_truth_id = filename_parts[0] if len(filename_parts) >= 1 else None
        has_ground_truth = ground_truth_id and len(ground_truth_id) >= 20 and ground_truth_id.replace('_', '').replace('-', '').isalnum()
        
        try:
            # Time the inference
            start_time = time.time()
            
            # Get top 5 matches (with optional OCR debug)
            ocr_debug_path = str(debug_dir / f"{img_path.stem}_ocr_debug.jpg") if debug_dir else None
            matches = identifier.identify(str(img_path), top_k=5, ocr_debug_path=ocr_debug_path)
            
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
            top = matches[0]
            print(f"✓ Top match: {top['name']}")
            print(f"  Printing ID: {top['printing_unique_id']}")
            print(f"  Combined Confidence: {top['combined_confidence']:.4f} (🔥 ENHANCED)")
            print(f"    ├─ Visual:  {top['visual_confidence']:.4f}")
            print(f"    ├─ pHash:   {top['phash_confidence']:.4f}")
            print(f"    └─ OCR:     {top['ocr_confidence']:.4f}")
            
            # OCR DEBUG INFO
            if top.get('ocr_debug'):
                print(f"\n  🔍 OCR Debug:")
                print(f"    Status: {top['ocr_debug']}")
                if top['ocr_text']:
                    print(f"    Extracted Text: \"{top['ocr_text']}\"")
                    
                    # Show top OCR matches from trained set (634 cards)
                    if top.get('ocr_top_matches'):
                        print(f"\n    Top 5 OCR Fuzzy Matches (trained set):")
                        for ocr_match in top['ocr_top_matches'][:5]:
                            print(f"      - {ocr_match['card_name']}: {ocr_match['ocr_score']:.3f}")
                    
                    # Show top OCR matches from FULL database (9000+ cards)
                    if top.get('full_db_ocr_matches'):
                        print(f"\n     Top 5 OCR Matches (FULL DATABASE - 9000+ cards):")
                        for card_id, card_name, score, _ in top['full_db_ocr_matches'][:5]:
                            print(f"      - {card_name}: {score:.3f} [{card_id[:15]}...]")
            
            if top.get('set_id'):
                print(f"\n  Set: {top['set_id']} | Edition: {top['edition']} | Foiling: {top['foiling']}")
            print(f"  Inference time: {inference_time*1000:.1f}ms")
            
            # Show top 5
            print(f"\n  Top 5 Combined Matches:")
            for match in matches:
                marker = " ←GT ✓" if has_ground_truth and match['printing_unique_id'] == ground_truth_id else ""
                print(f"    {match['rank']}. {match['name']} " +
                      f"(Combined: {match['combined_confidence']:.3f} | " +
                      f"Visual: {match['visual_confidence']:.3f} | " +
                      f"pHash: {match['phash_confidence']:.3f}){marker}")
            
            results.append({
                'image': img_path.name,
                'success': True,
                'top_match': matches[0],
                'inference_time_ms': inference_time * 1000,
                'has_ground_truth': has_ground_truth,
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
        
        print(f"\n Accuracy Metrics (on {total_with_gt} images with ground truth):")
        print(f"  Top-1 Accuracy: {top1_accuracy:.2f}% ({top1_correct}/{total_with_gt})")
        print(f"  Top-5 Accuracy: {top5_accuracy:.2f}% ({top5_correct}/{total_with_gt})")
    
    if successful > 0:
        avg_time = total_inference_time / successful * 1000
        print(f"\n⚡ Performance:")
        print(f"  Average inference time: {avg_time:.1f}ms")
        print(f"  Throughput: {successful/total_inference_time:.1f} images/second")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced multi-modal card identification")
    parser.add_argument('folder', type=str, help='Path to folder with test images')
    parser.add_argument('--model', type=str, default='small_scale_checkpoints/best_model.pth')
    parser.add_argument('--data-dir', type=str, default=r'D:\SIAMESE DATASET\SMALL SCALE OUTPUT')
    parser.add_argument('--json-db', type=str, default=r'..\fab_cards_data\card-flattened-with-phash.json',
                        help='Path to JSON database with card metadata')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild cache')
    parser.add_argument('--debug', action='store_true', help='Save OCR debug images showing bounding boxes')
    
    args = parser.parse_args()
    
    test_folder(args.folder, args.model, args.data_dir, args.json_db, args.rebuild, args.debug)
