"""
Training configuration constants.
Paths are passed as CLI arguments to train.py — nothing is hardcoded here.
"""

# Model architecture
EMBEDDING_DIM = 512
HIDDEN_DIM = 1024
BACKBONE = "mobilenetv3_large_100"
IMAGE_SIZE = 224

# Training
BATCH_SIZE = 64
NUM_EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 20          # early-stopping epochs without val-loss improvement
CHECK_INTERVAL = 5     # how often (in epochs) to compute top-k accuracy

# CosFace loss
COSFACE_SCALE = 30.0
COSFACE_MARGIN = 0.35

# DataLoader
NUM_WORKERS = 2
TRAIN_SPLIT = 0.85     # fraction of cards used for training
