"""
Model Configuration
"""

# Paths
IMAGE_DIR = "D:/SIAMESE DATASET/Images"
MODEL_SAVE_DIR = "model/checkpoints"
LOGS_DIR = "model/logs"

# Model Architecture
EMBEDDING_DIM = 512
BACKBONE = "mobilenetv3_small_100"  # timm model name
GEM_P_INIT = 3.0  # Initial GeM pooling parameter

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# ArcFace Loss
ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.5

# Data
IMAGE_SIZE = 224
NUM_WORKERS = 4

# Augmentation (already done in dataset prep, but can add more)
TRAIN_AUGMENTATION = True
