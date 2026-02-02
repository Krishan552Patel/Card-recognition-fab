# Recommended Improvements for Model Code

## CRITICAL FIXES

### 1. dataset.py - Fix Transform Issue (Line 208-218)

**Problem:** 
When using `random_split`, both train and val datasets reference the same underlying dataset object. Changing `val_dataset.dataset.transform` affects both splits.

**Fix:**
```python
# REPLACE create_dataloaders function with:
def create_dataloaders(root_dir, batch_size=32, num_workers=4, image_size=224, use_triplet=False):
    train_transform = get_transforms(image_size, augment=True)
    val_transform = get_transforms(image_size, augment=False)
    
    if use_triplet:
        train_dataset = TripletDataset(root_dir, transform=train_transform)
        val_dataset = TripletDataset(root_dir, transform=val_transform)
    else:
        # Create TWO separate dataset instances
        full_dataset_train = FABCardDataset(root_dir, transform=train_transform)
        full_dataset_val = FABCardDataset(root_dir, transform=val_transform)
        
        # Use subset instead of random_split
        indices = list(range(len(full_dataset_train)))
        train_size = int(0.8 * len(indices))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_dataset = torch.utils.data.Subset(full_dataset_train, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset_val, val_indices)
    
    # ... rest stays same
```

### 2. dataset.py - Remove RandomHorizontalFlip (Line 172)

**Problem:**
Cards have specific orientations! Horizontal flip would create invalid images.

**Fix:**
```python
# Remove this line:
# transforms.RandomHorizontalFlip(p=0.5),  # DON'T FLIP CARDS!
```

### 3. train.py - Add Gradient Clipping

**Why:** Prevents exploding gradients, especially with ArcFace loss.

**Add after line 37:**
```python
# Backward pass
optimizer.zero_grad()
loss.backward()

# ADD THIS:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

optimizer.step()
```

### 4. train.py - Add Mixed Precision Training

**Why:** 2-3x faster training on modern GPUs with minimal quality loss.

**Add at top:**
```python
from torch.cuda.amp import autocast, GradScaler
```

**Modify train function:**
```python
# After creating optimizer (line 132):
scaler = GradScaler() if device.type == 'cuda' else None

# In train_epoch, replace forward/backward with:
with autocast(enabled=scaler is not None):
    embeddings = model(images)
    loss = criterion(embeddings, labels)

optimizer.zero_grad()
if scaler:
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
```

### 5. train.py - Add Accuracy Metrics

**Add function:**
```python
def compute_accuracy(model, dataloader, device):
    """Compute top-1 and top-5 accuracy."""
    model.eval()
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            embeddings = model(images)
            
            # Cosine similarity to all classes (using ArcFace weights)
            # This is simplified - you'd need access to criterion.weight
            
            total += labels.size(0)
    
    return correct_1 / total, correct_5 / total
```

---

## RECOMMENDED IMPROVEMENTS

### 6. backbone.py - Add Dropout

**Add after line 101:**
```python
# Batch normalization
self.bn = nn.BatchNorm1d(embedding_dim)

# ADD THIS:
self.dropout = nn.Dropout(p=0.2)  # Regularization
```

**Update forward (line 119):**
```python
embedding = self.fc(pooled)
embedding = self.bn(embedding)
embedding = self.dropout(embedding)  # ADD THIS
embedding = F.normalize(embedding, p=2, dim=1)
```

### 7. backbone.py - Add Freeze Option

**Update __init__:**
```python
def __init__(self, embedding_dim=512, gem_p=3.0, pretrained=True, freeze_backbone=False):
    super(CardEmbeddingNet, self).__init__()
    
    # ... existing code ...
    
    # ADD THIS:
    if freeze_backbone:
        for param in self.backbone.parameters():
            param.requires_grad = False
```

### 8. train.py - Add Resume Capability

**Add argument:**
```python
parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')

# In train function:
start_epoch = 1
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {checkpoint['epoch']}")

# Update loop:
for epoch in range(start_epoch, num_epochs + 1):
```

### 9. config.py - Add More Options

```python
# Training stability
GRADIENT_CLIP_NORM = 5.0
USE_MIXED_PRECISION = True  # Requires CUDA
DROPOUT_RATE = 0.2

# Backbone
FREEZE_BACKBONE_EPOCHS = 5  # Freeze first N epochs

# Logging
LOG_INTERVAL = 10  # Log every N batches
SAVE_INTERVAL = 5  # Save checkpoint every N epochs
```

---

## PRIORITY FIXES

**Must fix before training:**
1. ✅ Dataset transform bug (CRITICAL)
2. ✅ Remove horizontal flip (CRITICAL)
3. ✅ Add gradient clipping (HIGH)

**Recommended for better results:**
4. ⚠️ Mixed precision training (if using GPU)
5. ⚠️ Dropout in backbone
6. ⚠️ Resume capability
7. ⚠️ Accuracy metrics

**Nice to have:**
8. Freeze backbone option
9. Better logging
10. Early stopping

Would you like me to apply these fixes to your code?
