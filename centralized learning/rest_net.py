import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import defaultdict
import time

# Set plotting style
plt.style.use('ggplot')
sns.set_palette("husl")

# Configuration
DATA_DIR = '/home/oussama/Desktop/MLA2/centralized learning/unified_dataset'
METADATA_CSV = os.path.join(DATA_DIR, 'dataset_metadata.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 70
WARMUP_EPOCHS = 5
IMG_SIZE = 32
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXCLUDE_CLASS = 'HamzaI'
NUM_CLASSES = 28  # Arabic letters without hamza

# Set random seeds
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class UnifiedArabicDataset(Dataset):
    """Dataset class for Arabic letters with Hamza exclusion"""
    def __init__(self, metadata_csv, images_dir, transform=None, exclude_class=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
        
        self.metadata = pd.read_csv(metadata_csv)
        
        # Exclude hamza class
        if exclude_class:
            original_count = len(self.metadata)
            self.metadata = self.metadata[self.metadata['label'] != exclude_class]
            excluded_count = original_count - len(self.metadata)
            print(f"✓ Excluded '{exclude_class}': {excluded_count} images removed")
        
        if len(self.metadata) == 0:
            raise ValueError(f"No data found in {metadata_csv}")
        
        # Get classes and create mappings (reindex after exclusion)
        self.classes = sorted(self.metadata['label'].unique().tolist())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        self.datasets = []
        
        for _, row in self.metadata.iterrows():
            img_path = self.images_dir / row['filename']
            if img_path.exists():
                self.image_paths.append(str(img_path))
                self.labels.append(self.class_to_idx[row['label']])
                self.datasets.append(row['dataset'])
        
        print(f"✓ Loaded {len(self.image_paths)} images across {len(self.classes)} classes")
        
        # Dataset statistics
        dataset_counts = {}
        for ds in self.datasets:
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        
        print("\nDistribution by Source:")
        for ds_name in sorted(dataset_counts.keys()):
            print(f"  {ds_name}: {dataset_counts[ds_name]} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ==================== RESNET BUILDING BLOCKS ====================

class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet
    
    Structure:
    x → Conv3x3 → BN → ReLU → Conv3x3 → BN → (+) → ReLU
    └──────────────────────────────────────────┘
                    (shortcut)
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add shortcut and apply ReLU
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18_32x32(nn.Module):
    """
    ResNet-18 Architecture adapted for 32x32 Grayscale Images
    
    Optimized for localized Arabic letter images:
    - Modified initial conv (3x3 instead of 7x7, stride=1)
    - No initial maxpool (preserves spatial resolution)
    - 4 residual layer groups with [2, 2, 2, 2] blocks
    - Minimal downsampling for 32x32 input
    
    Architecture:
    Input: 32x32x1 (grayscale)
    
    Initial Conv: 32x32x1 → 32x32x64 (stride=1, NO pooling)
    
    Layer 1 [64]:  32x32x64  → 32x32x64  (2 blocks, NO downsampling)
    Layer 2 [128]: 32x32x64  → 16x16x128 (2 blocks, stride=2)
    Layer 3 [256]: 16x16x128 → 8x8x256   (2 blocks, stride=2)
    Layer 4 [512]: 8x8x256   → 4x4x512   (2 blocks, stride=2)
    
    Global Avg Pool: 4x4x512 → 1x1x512
    FC: 512 → 28 classes
    
    Total Parameters: ~11M
    """
    
    def __init__(self, num_classes=28):
        super(ResNet18_32x32, self).__init__()
        
        self.in_channels = 64
        
        # ===== INITIAL CONVOLUTION (Modified for 32x32) =====
        # Standard ResNet uses 7x7 kernel with stride=2 and maxpool
        # For 32x32, we use 3x3 kernel with stride=1 and NO maxpool
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # NO maxpool to preserve spatial resolution!
        
        # ===== RESIDUAL LAYERS =====
        # ResNet-18 uses [2, 2, 2, 2] blocks
        self.layer1 = self._make_layer(64, 2, stride=1)   # 32x32x64
        self.layer2 = self._make_layer(128, 2, stride=2)  # 16x16x128
        self.layer3 = self._make_layer(256, 2, stride=2)  # 8x8x256
        self.layer4 = self._make_layer(512, 2, stride=2)  # 4x4x512
        
        # ===== GLOBAL AVERAGE POOLING =====
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ===== CLASSIFIER =====
        self.fc = nn.Linear(512, num_classes)
        
        # ===== WEIGHT INITIALIZATION =====
        self._initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride):
        """Create a residual layer with multiple blocks"""
        downsample = None
        
        # If stride != 1 or channels change, we need a downsample layer for shortcut
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        
        # First block (may downsample)
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)    # 32x32x64
        x = self.bn1(x)
        x = self.relu(x)
        # No maxpool!
        
        # Residual layers
        x = self.layer1(x)   # 32x32x64
        x = self.layer2(x)   # 16x16x128
        x = self.layer3(x)   # 8x8x256
        x = self.layer4(x)   # 4x4x512
        
        # Global average pooling
        x = self.avgpool(x)  # 1x1x512
        x = torch.flatten(x, 1)  # 512
        
        # Classification
        x = self.fc(x)       # num_classes
        
        return x


class ResNet34_32x32(nn.Module):
    """
    ResNet-34 Architecture adapted for 32x32 Grayscale Images
    
    Deeper than ResNet-18 with [3, 4, 6, 3] blocks
    Better feature learning for complex Arabic letter patterns
    
    Total Parameters: ~21M
    """
    
    def __init__(self, num_classes=28):
        super(ResNet34_32x32, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution (modified for 32x32)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual layers - ResNet-34 uses [3, 4, 6, 3] blocks
        self.layer1 = self._make_layer(64, 3, stride=1)   # 32x32x64
        self.layer2 = self._make_layer(128, 4, stride=2)  # 16x16x128
        self.layer3 = self._make_layer(256, 6, stride=2)  # 8x8x256
        self.layer4 = self._make_layer(512, 3, stride=2)  # 4x4x512
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(512, num_classes)
        
        # Weight initialization
        self._initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ==================== TRAINING FUNCTIONS ====================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                warmup_scheduler, epochs=70, warmup_epochs=5):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)
    print(f"Warmup: {warmup_epochs} epochs | Cosine: {epochs - warmup_epochs} epochs")
    print(f"Early stopping patience: {patience}")
    print("="*70)
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # LR scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
            sched_name = "Warmup"
        else:
            scheduler.step()
            sched_name = "Cosine"
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"✓ Best: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"E[{epoch+1:02d}/{epochs}] {sched_name:6s} | "
              f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | "
              f"Val: {val_loss:.4f} {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if patience_counter >= patience:
            print(f"\n⚠ Early stop at epoch {epoch+1}")
            break
    
    print("="*70)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print("="*70 + "\n")
    
    return train_losses, train_accs, val_losses, val_accs


def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc


def plot_metrics(train_losses, train_accs, val_losses, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs_range, train_losses, 'o-', label='Train', linewidth=2, markersize=4)
    axes[0].plot(epochs_range, val_losses, 's-', label='Val', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Loss', fontweight='bold', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(epochs_range, train_accs, 'o-', label='Train', linewidth=2, markersize=4)
    axes[1].plot(epochs_range, val_accs, 's-', label='Val', linewidth=2, markersize=4)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[1].set_title('Accuracy', fontweight='bold', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resnet_training.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: resnet_training.png")
    plt.close()


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(14, 12))
    
    annotations = []
    for i in range(len(class_names)):
        row = []
        row_sum = cm[i].sum()
        for j in range(len(class_names)):
            count = cm[i, j]
            if row_sum > 0:
                pct = (count / row_sum) * 100
                row.append(f'{count}\n{pct:.1f}%' if count > 0 else '')
            else:
                row.append('')
        annotations.append(row)
    
    sns.heatmap(cm, annot=np.array(annotations), fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted', fontweight='bold', fontsize=12)
    ax.set_ylabel('True', fontweight='bold', fontsize=12)
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('resnet_confusion.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: resnet_confusion.png")
    plt.close()


def analyze_results(all_labels, all_preds, class_names):
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    correct = sum(1 for t, p in zip(all_labels, all_preds) if t == p)
    total = len(all_labels)
    acc = 100 * correct / total
    
    print(f"\nAccuracy: {acc:.2f}% ({correct}/{total})")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, cls in enumerate(class_names):
        mask = [l == i for l in all_labels]
        if sum(mask) > 0:
            cls_correct = sum(1 for t, p, m in zip(all_labels, all_preds, mask) 
                            if m and t == p)
            cls_total = sum(mask)
            cls_acc = 100 * cls_correct / cls_total
            print(f"  {cls:15s}: {cls_acc:5.1f}% ({cls_correct}/{cls_total})")
    
    print("="*80)


def main():
    print("\n" + "="*80)
    print("RESNET FOR 32x32 GRAYSCALE ARABIC LETTERS (28 CLASSES - NO HAMZA)")
    print("="*80)
    print("Architecture Options:")
    print("  1. ResNet-18: 2-2-2-2 blocks (~11M params)")
    print("  2. ResNet-34: 3-4-6-3 blocks (~21M params)")
    print("="*80 + "\n")
    
    # Choose architecture
    print("Select ResNet variant:")
    print("  [1] ResNet-18 (lighter, faster)")
    print("  [2] ResNet-34 (deeper, more capacity)")
    choice = input("Enter choice (1 or 2) [default=1]: ").strip() or "1"
    
    print(f"\nDevice: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Load dataset
    try:
        full_dataset = UnifiedArabicDataset(METADATA_CSV, IMAGES_DIR, 
                                           transform=None, exclude_class=EXCLUDE_CLASS)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return
    
    # Verify class count
    actual_classes = len(full_dataset.classes)
    print(f"\n✓ Actual classes in dataset: {actual_classes}")
    if actual_classes != NUM_CLASSES:
        print(f"⚠ Warning: Expected {NUM_CLASSES} classes, found {actual_classes}")
    
    # Split
    indices = list(range(len(full_dataset)))
    total_size = len(indices)
    test_size = int(TEST_SPLIT * total_size)
    val_size = int(VAL_SPLIT * total_size)
    train_size = total_size - test_size - val_size
    
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"\nSplit: Train={len(train_indices)} Val={len(val_indices)} Test={len(test_indices)}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(12),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Datasets
    train_dataset_full = UnifiedArabicDataset(METADATA_CSV, IMAGES_DIR, 
                                             train_transform, EXCLUDE_CLASS)
    val_dataset_full = UnifiedArabicDataset(METADATA_CSV, IMAGES_DIR, 
                                           eval_transform, EXCLUDE_CLASS)
    test_dataset_full = UnifiedArabicDataset(METADATA_CSV, IMAGES_DIR, 
                                            eval_transform, EXCLUDE_CLASS)
    
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    test_dataset = Subset(test_dataset_full, test_indices)
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    if choice == "2":
        model = ResNet34_32x32(num_classes=actual_classes).to(DEVICE)
        model_name = "ResNet-34"
    else:
        model = ResNet18_32x32(num_classes=actual_classes).to(DEVICE)
        model_name = "ResNet-18"
    
    print(f"\n{'='*70}")
    print(f"{model_name} Architecture:")
    print(f"{'='*70}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print(f"{'='*70}\n")
    
    # Training
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, WARMUP_EPOCHS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS - WARMUP_EPOCHS, 1e-6)
    
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        cosine_scheduler, warmup_scheduler, EPOCHS, WARMUP_EPOCHS
    )
    
    plot_metrics(train_losses, train_accs, val_losses, val_accs)
    
    # Evaluate
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=full_dataset.classes, digits=4))
    
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, full_dataset.classes)
    analyze_results(all_labels, all_preds, full_dataset.classes)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Classes: {actual_classes} (hamza excluded)")
    print(f"Parameters: {params:,}")
    print("Files: resnet_training.png, resnet_confusion.png")
    print("="*80)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTime: {time.time() - start:.1f}s")