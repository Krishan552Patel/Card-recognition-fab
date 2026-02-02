# 🚀 Google Colab Training Guide

## Quick Start

### Step 1: Prepare Your Data

**Option A: Upload to Google Drive (Recommended)**
1. Create a folder in your Google Drive: `CardImages`
2. Inside, create subfolders for each card with the card ID as name:
   ```
   MyDrive/
   └── CardImages/
       ├── ELE001-RF/
       │   └── image.png
       ├── ELE002-CF/
       │   └── image.png
       └── ...
   ```
3. Each folder = 1 card, can have multiple images

**Option B: Upload as ZIP**
1. Zip your `Images` folder
2. Upload directly in Colab (< 1GB recommended)

---

### Step 2: Open the Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `Card_Recognition_Training.ipynb`
4. Or: **File → Open notebook → GitHub/Drive**

---

### Step 3: Enable GPU

1. Click **Runtime → Change runtime type**
2. Select **T4 GPU** (free tier) or **V100** (if available)
3. Click **Save**

---

### Step 4: Run the Notebook

Run cells in order:
1. **Setup** - Install packages, mount Drive
2. **Upload Data** - Configure your data path
3. **Model** - Creates the color-aware neural network
4. **Training** - Trains with early stopping
5. **Export** - Saves ONNX for Jetson Nano

---

## Training Configuration

Edit the `CONFIG` cell to adjust:

```python
CONFIG = {
    'epochs': 100,        # Max epochs (early stopping may stop earlier)
    'batch_size': 64,     # Reduce if OOM error
    'learning_rate': 1e-3,
    'patience': 15,       # Early stopping patience
}
```

---

## Expected Training Time

| GPU | Cards | Time |
|-----|-------|------|
| T4 (free) | 100 | ~30 min |
| T4 (free) | 1000 | ~3 hours |
| V100 | 1000 | ~1 hour |

---

## Output Files

After training, you'll have in Google Drive:
- `best_model.pth` - PyTorch checkpoint
- `card_recognition.onnx` - ONNX model for Jetson
- `training_curves.png` - Loss visualization

---

## Deploy to Jetson Nano

1. Copy `card_recognition.onnx` to Jetson
2. Convert to TensorRT:
   ```bash
   /usr/src/tensorrt/bin/trtexec \
       --onnx=card_recognition.onnx \
       --saveEngine=card_recognition.engine \
       --fp16
   ```
3. Use the `.engine` file for inference (~50 FPS)

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` to 32 or 16
- Restart runtime (**Runtime → Restart runtime**)

### "Drive not mounting"
- Make sure you're signed into the correct Google account
- Try: **Runtime → Restart runtime**, then remount

### "Training loss not decreasing"
- Check your data structure matches expected format
- Try reducing learning rate to `5e-4`

---

## Tips for Best Accuracy

1. **More cards = better** - Try to have 500+ unique cards
2. **Multiple images per card** - 4-8 images per card helps
3. **Include rotations** - 0°, 90°, 180°, 270° in training data
4. **Watch for collapse** - If "mean similarity" goes > 0.9, something's wrong
