"""
Test Inference on Multiple Images
Tests the card identifier on all images in a folder.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from inference import CardIdentifier


def test_folder(folder_path: str, model_path: str = 'model/checkpoints/best_model.pth', force_rebuild: bool = False):
    """
    Test card identification on all images in a folder.
    
    Args:
        folder_path: Path to folder containing test images
        model_path: Path to trained model
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
    identifier = CardIdentifier(model_path=model_path, force_rebuild=force_rebuild)
    
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
        # Format: <printing_unique_id>.jpg or <printing_unique_id>.<anything>.jpg
        filename_parts = img_path.stem.split('.')
        ground_truth_id = filename_parts[0] if len(filename_parts) >= 1 else None
        
        # Check if this looks like a valid printing_id (21-22 character alphanumeric)
        has_ground_truth = ground_truth_id and len(ground_truth_id) >= 20 and ground_truth_id.replace('_', '').replace('-', '').isalnum()
        
        try:
            # Time the inference
            import time
            start_time = time.time()
            
            # Get top 5 matches
            matches = identifier.identify(str(img_path), top_k=5, return_details=True)
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Validate if we have ground truth
            is_top1_correct = False
            is_top5_correct = False
            
            if has_ground_truth:
                total_with_gt += 1
                
                # Check Top-1 exact match
                if matches[0]['printing_unique_id'] == ground_truth_id:
                    is_top1_correct = True
                    is_top5_correct = True
                    top1_correct += 1
                    top5_correct += 1
                else:
                    # Check Top-5 with pHash matching
                    # Get ground truth card's pHash from database
                    gt_card = identifier.db.get_card_by_printing_id(ground_truth_id)
                    
                    if gt_card and gt_card.get('image_phash'):
                        # Check if any top-5 candidate shares the same pHash
                        for match in matches:
                            pred_card = identifier.db.get_card_by_printing_id(match['printing_unique_id'])
                            
                            if pred_card and pred_card.get('image_phash') == gt_card.get('image_phash'):
                                is_top5_correct = True
                                top5_correct += 1
                                break
            
            # Display results (without ground truth status)
            print(f"Top match: {matches[0]['name']}")
            print(f"  Printing ID: {matches[0]['printing_unique_id']}")
            print(f"  Card ID: {matches[0].get('card_id', 'N/A')}")
            print(f"  Set: {matches[0].get('set_id', 'N/A')} | Edition: {matches[0].get('edition', 'N/A')}")
            print(f"  Foiling: {matches[0].get('foiling', 'N/A')} | Rarity: {matches[0].get('rarity', 'N/A')}")
            
            # Show pitch/cost if available
            details = []
            if matches[0].get('pitch'):
                details.append(f"Pitch: {matches[0]['pitch']}")
            if matches[0].get('cost'):
                details.append(f"Cost: {matches[0]['cost']}")
            if matches[0].get('power'):
                details.append(f"Power: {matches[0]['power']}")
            if matches[0].get('defense'):
                details.append(f"Defense: {matches[0]['defense']}")
            if details:
                print(f"  {' | '.join(details)}")
            
            print(f"  Confidence: {matches[0]['visual_confidence']:.4f}")
            print(f"  Inference time: {inference_time*1000:.1f}ms")
            
            # Show detailed top 5
            print(f"\n  Top 5 candidates:")
            for match in matches:
                # Build compact info string
                info_parts = [match['name']]
                if match.get('card_id'):
                    info_parts.append(f"ID:{match['card_id']}")
                if match.get('set_id'):
                    info_parts.append(f"Set:{match['set_id']}")
                if match.get('edition'):
                    info_parts.append(f"Ed:{match['edition']}")
                if match.get('foiling'):
                    info_parts.append(f"Foil:{match['foiling']}")
                if match.get('rarity'):
                    info_parts.append(f"Rare:{match['rarity']}")
                if match.get('pitch'):
                    info_parts.append(f"P:{match['pitch']}")
                
                info_str = " | ".join(info_parts)
                
                # Highlight if this matches ground truth
                marker = " ←GT" if has_ground_truth and match['printing_unique_id'] == ground_truth_id else ""
                print(f"    {match['rank']}. {info_str} ({match['visual_confidence']:.3f}){marker}")
            
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
                'error': str(e),
                'has_ground_truth': has_ground_truth,
                'ground_truth_id': ground_truth_id if has_ground_truth else None
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results if r['success'])
    print(f"Successfully identified: {successful}/{len(results)}")
    
    # Accuracy metrics (only for images with ground truth)
    if total_with_gt > 0:
        top1_accuracy = (top1_correct / total_with_gt) * 100
        top5_accuracy = (top5_correct / total_with_gt) * 100
        
        print(f"\nAccuracy Metrics (on {total_with_gt} images with ground truth):")
        print(f"  Top-1 Accuracy (exact match): {top1_accuracy:.2f}% ({top1_correct}/{total_with_gt})")
        print(f"  Top-5 Accuracy (with pHash): {top5_accuracy:.2f}% ({top5_correct}/{total_with_gt})")
    if successful > 0:
        avg_time = total_inference_time / successful * 1000
        print(f"\nPerformance:")
        print(f"  Total inference time: {total_inference_time:.2f}s")
        print(f"  Average per image: {avg_time:.1f}ms")
        print(f"  Throughput: {successful/total_inference_time:.1f} images/second")
    
    if successful < len(results):
        print(f"\nFailed images:")
        for r in results:
            if not r['success']:
                print(f"  - {r['image']}: {r['error']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test card identification on folder")
    parser.add_argument('folder', type=str, help='Path to folder with test images')
    parser.add_argument('--model', type=str, default='model/checkpoints/best_model.pth')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild embeddings cache')
    
    args = parser.parse_args()
    
    # Pass force_rebuild to CardIdentifier
    identifier = CardIdentifier(model_path=args.model, force_rebuild=args.rebuild)
    
    test_folder(args.folder, args.model, args.rebuild)
