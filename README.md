# Card Recognition Model

Training pipeline for card game identification using Siamese networks.

## 🚀 Quick Start with Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/Card_Recognition_Training.ipynb)

1. Click the badge above (after pushing to GitHub)
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run all cells

## 📁 Repository Structure

```
model/
├── Card_Recognition_Training.ipynb  # Main Colab notebook
├── COLAB_GUIDE.md                   # Training guide
├── backbone.py                      # Model architecture
├── arcface_loss.py                  # Loss functions
├── dataset.py                       # Data loading
├── train.py                         # Training script
├── inference.py                     # Inference pipeline
├── config.py                        # Configuration
├── evaluate.py                      # Evaluation
└── checkpoints/                     # Saved models (gitignored)
```

## 📦 Your Card Data

**DO NOT upload card images to GitHub!** (too large + copyright)

Your data is at:
```
D:\SIAMESE DATASET\LARGE SCALE OUTPUT\
├── 66gcPMPfHnNqfQQqz8PCL\    ← Card ID folder
│   ├── rot_000.png           ← Original
│   ├── rot_090.png           ← 90° rotation
│   ├── rot_180.png           ← 180° rotation
│   └── rot_270.png           ← 270° rotation
├── 66GnLh6pGrGFpKzckjkhp\
│   └── ...
└── ... (13,948 cards total)
```

**Upload to Google Drive** as a ZIP:
1. Zip `D:\SIAMESE DATASET\LARGE SCALE OUTPUT` → `card_images.zip`
2. Upload to `MyDrive/CardData/card_images.zip`
3. Colab will unzip automatically

## 🔧 Features

- **Color-Aware Model**: Detects similar cards with different colors
- **CosFace Loss**: More stable than ArcFace for card recognition
- **Sim-to-Real Augmentation**: Train on scans, deploy on camera
- **Jetson Nano Optimized**: ONNX export for TensorRT conversion

## 📊 Expected Accuracy

| Cards | Top-1 | Top-5 |
|-------|-------|-------|
| 100   | ~90%  | 99%   |
| 1000  | ~95%  | 99%   |
| 10000 | ~97%  | 99%   |

## 🖥️ Deployment

After training, export to Jetson Nano:
```bash
# On Jetson Nano
/usr/src/tensorrt/bin/trtexec \
    --onnx=card_recognition.onnx \
    --saveEngine=card_recognition.engine \
    --fp16
```
