import os
import torch
import torch.nn as nn
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

# Configuration - UPDATED FOR UNIFIED DATASET
DATA_DIR = '/home/oussama/Desktop/MLA2/centralized learning/unified_dataset'
METADATA_CSV = os.path.join(DATA_DIR, 'dataset_metadata.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
BATCH_SIZE = 64  # Increased for better GPU utilization
LEARNING_RATE = 0.001
EPOCHS = 50
WARMUP_EPOCHS = 3
IMG_SIZE = 32
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Enable cuDNN benchmarking for faster training
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

class UnifiedArabicDataset(Dataset):
    """
    Dataset class for the unified Arabic letter dataset
    Reads from metadata CSV file
    """
    def __init__(self, metadata_csv, images_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Load metadata
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
        
        self.metadata = pd.read_csv(metadata_csv)
        
        if len(self.metadata) == 0:
            raise ValueError(f"No data found in {metadata_csv}")
        
        # Get classes and create mappings
        self.classes = sorted(self.metadata['label'].unique().tolist())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Prepare image paths and labels
        self.image_paths = []
        self.labels = []
        self.datasets = []
        
        for _, row in self.metadata.iterrows():
            img_path = self.images_dir / row['filename']
            if img_path.exists():
                self.image_paths.append(str(img_path))
                self.labels.append(row['label_idx'])
                self.datasets.append(row['dataset'])
            else:
                print(f"Warning: Image not found: {img_path}")
        
        # Print dataset statistics
        print(f"\nLoaded {len(self.image_paths)} images across {len(self.classes)} classes.")
        print(f"Classes: {self.classes}")
        
        # Print distribution by source dataset
        dataset_counts = {}
        for ds in self.datasets:
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        
        print("\nDistribution by Source Dataset:")
        for ds_name in sorted(dataset_counts.keys()):
            print(f"  {ds_name}: {dataset_counts[ds_name]} images")
        
        # Print class distribution
        class_counts = {}
        for label in self.labels:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\nClass Distribution:")
        for cls_name in sorted(class_counts.keys()):
            print(f"  {cls_name}: {class_counts[cls_name]} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class SlightlyEnhancedCNN(nn.Module):
    """
    Slightly deeper CNN for 32x32 grayscale images.
    - No downsampling (stride=1 everywhere)
    - More convolutional layers for richer feature extraction
    - Global average pooling for spatial invariance
    - Small classifier at the end
    """
    def __init__(self, num_classes):
        super(SlightlyEnhancedCNN, self).__init__()
        
        # Block 1: 32x32x1 -> 32x32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Block 2: 32x32x32 -> 32x32x64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Block 3: 32x32x64 -> 32x32x128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Block 4: 32x32x128 -> 32x32x256
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling: 32x32x256 -> 1x1x256
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier: 256 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)  # 32x32x32
        x = self.block2(x)  # 32x32x64
        x = self.block3(x)  # 32x32x128
        x = self.block4(x)  # 32x32x256
        x = self.avgpool(x) # 1x1x256
        x = torch.flatten(x, 1)  # 256
        x = self.classifier(x)  # num_classes
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, warmup_scheduler, epochs=50, warmup_epochs=3):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    print("\n" + "="*60)
    print("Starting training...")
    print(f"Warmup: {warmup_epochs} epochs | Cosine Annealing: {epochs - warmup_epochs} epochs")
    print(f"Early stopping patience: {patience} epochs")
    print("="*60)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
            scheduler_name = "Warmup"
        else:
            scheduler.step()
            scheduler_name = "Cosine"
        
        epoch_time = time.time() - epoch_start
        
        # Save best model and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'best_model_enhanced.pth')
            print(f"✓ New best model saved (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] [{scheduler_name}] | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
            break
    
    print("="*60)
    print(f"Training completed! Best Val Accuracy: {best_val_acc:.2f}%")
    print("="*60 + "\n")
              
    return train_losses, train_accs, val_losses, val_accs


def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
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
    
    # Loss Plot
    axes[0].plot(epochs_range, train_losses, marker='o', linewidth=2, label='Train Loss', markersize=6)
    axes[0].plot(epochs_range, val_losses, marker='s', linewidth=2, label='Val Loss', markersize=6)
    axes[0].set_xlabel('Epochs', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy Plot
    axes[1].plot(epochs_range, train_accs, marker='o', linewidth=2, label='Train Acc', markersize=6)
    axes[1].plot(epochs_range, val_accs, marker='s', linewidth=2, label='Val Acc', markersize=6)
    axes[1].set_xlabel('Epochs', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics_enhanced.png', dpi=300, bbox_inches='tight')
    print("✓ Metrics plot saved to 'training_metrics_enhanced.png'")
    plt.close()


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix with percentages"""
    fig, ax = plt.subplots(figsize=(max(12, len(class_names)), max(10, len(class_names) * 0.8)))
    
    # Create annotations with counts and percentages
    annotations = []
    for i in range(len(class_names)):
        row = []
        row_sum = cm[i].sum()
        for j in range(len(class_names)):
            count = cm[i, j]
            if row_sum > 0:
                percentage = (count / row_sum) * 100
                if count > 0:
                    row.append(f'{count}\n({percentage:.1f}%)')
                else:
                    row.append('')
            else:
                row.append(str(count) if count > 0 else '')
        annotations.append(row)
    
    # Create heatmap
    sns.heatmap(cm, annot=np.array(annotations), fmt='', cmap='YlGnBu', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=0.5, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Count and Percentage)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_enhanced.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to 'confusion_matrix_enhanced.png'")
    plt.close()


def analyze_misclassifications(all_labels, all_preds, class_names):
    """Analyze and print detailed misclassification information"""
    print("\n" + "="*80)
    print("DETAILED MISCLASSIFICATION ANALYSIS")
    print("="*80)
    
    # Store misclassifications
    misclass_dict = defaultdict(lambda: defaultdict(int))
    correct_count = 0
    total_count = len(all_labels)
    
    for true_label, pred_label in zip(all_labels, all_preds):
        if true_label == pred_label:
            correct_count += 1
        else:
            true_class = class_names[true_label]
            pred_class = class_names[pred_label]
            misclass_dict[true_class][pred_class] += 1
    
    # Print overall accuracy
    accuracy = (correct_count / total_count) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    print(f"Total Misclassifications: {total_count - correct_count}")
    
    # Print misclassifications for each class
    if len(misclass_dict) > 0:
        print("\n" + "-"*80)
        print("Misclassifications by True Label:")
        print("-"*80)
        
        for true_class in sorted(misclass_dict.keys()):
            predictions = misclass_dict[true_class]
            total_misclass = sum(predictions.values())
            
            print(f"\n'{true_class}' was misclassified {total_misclass} times:")
            
            # Sort by frequency
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            for pred_class, count in sorted_preds[:5]:  # Show top 5 misclassifications
                print(f"  → Predicted as '{pred_class}': {count} times")
    else:
        print("\n🎉 No misclassifications! Perfect accuracy!")
    
    print("\n" + "="*80)


def plot_per_class_accuracy(all_labels, all_preds, class_names):
    """Plot per-class accuracy bar chart"""
    from sklearn.metrics import accuracy_score
    
    # Calculate per-class accuracy
    class_accuracies = []
    class_counts = []
    
    for i, class_name in enumerate(class_names):
        # Get indices where true label is this class
        class_mask = np.array(all_labels) == i
        class_true = np.array(all_labels)[class_mask]
        class_pred = np.array(all_preds)[class_mask]
        
        if len(class_true) > 0:
            acc = accuracy_score(class_true, class_pred) * 100
            class_accuracies.append(acc)
            class_counts.append(len(class_true))
        else:
            class_accuracies.append(0)
            class_counts.append(0)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.6), 6))
    
    bars = ax.bar(range(len(class_names)), class_accuracies, color='steelblue', alpha=0.8)
    
    # Color bars based on accuracy
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        if acc >= 90:
            bar.set_color('green')
        elif acc >= 70:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Add value labels on bars
    for i, (acc, count) in enumerate(zip(class_accuracies, class_counts)):
        ax.text(i, acc + 1, f'{acc:.1f}%\n(n={count})', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy on Test Set', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='≥90% (Excellent)'),
        Patch(facecolor='orange', label='70-89% (Good)'),
        Patch(facecolor='red', label='<70% (Needs Improvement)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('per_class_accuracy_enhanced.png', dpi=300, bbox_inches='tight')
    print("✓ Per-class accuracy plot saved to 'per_class_accuracy_enhanced.png'")
    plt.close()


def main():
    print("\n" + "="*80)
    print("ENHANCED CNN - UNIFIED ARABIC LETTER TRAINING")
    print("="*80)
    print(f"Architecture: SlightlyEnhancedCNN (4 conv blocks, no pooling)")
    print(f"Image size: 32x32 grayscale")
    print(f"Dataset: {DATA_DIR}")
    print("="*80 + "\n")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Benchmark: Enabled")
    print()
    
    # Load dataset
    try:
        full_dataset = UnifiedArabicDataset(METADATA_CSV, IMAGES_DIR, transform=None)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # Create train/val/test indices
    indices = list(range(len(full_dataset)))
    
    # Calculate split sizes
    total_size = len(indices)
    test_size = int(TEST_SPLIT * total_size)
    val_size = int(VAL_SPLIT * total_size)
    train_size = total_size - test_size - val_size
    
    # Set seed and shuffle indices
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"\n{'='*60}")
    print(f"Dataset Split:")
    print(f"  Training set: {len(train_indices)} images ({100*train_size/total_size:.1f}%)")
    print(f"  Validation set: {len(val_indices)} images ({100*val_size/total_size:.1f}%)")
    print(f"  Test set: {len(test_indices)} images ({100*test_size/total_size:.1f}%)")
    print(f"  Total: {total_size} images")
    print(f"{'='*60}\n")
    
    # Training transform with augmentation (NO horizontal flip)
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            shear=5
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Validation/Test transform (NO augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create separate dataset instances
    train_dataset_full = UnifiedArabicDataset(METADATA_CSV, IMAGES_DIR, transform=train_transform)
    val_dataset_full = UnifiedArabicDataset(METADATA_CSV, IMAGES_DIR, transform=eval_transform)
    test_dataset_full = UnifiedArabicDataset(METADATA_CSV, IMAGES_DIR, transform=eval_transform)
    
    # Apply the split using Subset
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    test_dataset = Subset(test_dataset_full, test_indices)
    
    print("✓ Augmentation applied ONLY to training set")
    print("✓ NO horizontal flip - preserves Arabic letter orientation\n")
    
    # DataLoaders with optimized settings
    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Model Setup
    num_classes = len(full_dataset.classes)
    model = SlightlyEnhancedCNN(num_classes).to(DEVICE)
    
    print("Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Linear warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS
    )
    
    # Cosine annealing scheduler
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS - WARMUP_EPOCHS,
        eta_min=1e-6
    )
    
    # Training
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        cosine_scheduler, warmup_scheduler, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS
    )
    
    # Load best model
    checkpoint = torch.load('best_model_enhanced.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Loaded best model for final evaluation\n")
    
    # Plotting training metrics
    plot_metrics(train_losses, train_accs, val_losses, val_accs)
    
    # Final Evaluation on Test Set
    print("="*80)
    print("Final Evaluation on Test Set")
    print("="*80)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, full_dataset.classes)
    
    # Per-class accuracy plot
    plot_per_class_accuracy(all_labels, all_preds, full_dataset.classes)
    
    # Detailed misclassification analysis
    analyze_misclassifications(all_labels, all_preds, full_dataset.classes)
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    print("Model: SlightlyEnhancedCNN")
    print("  - 4 convolutional blocks (32->64->128->256 channels)")
    print("  - No spatial downsampling (preserves 32x32 resolution)")
    print("  - Global Average Pooling for spatial invariance")
    print("  - Optimized for GPU performance")
    print(f"  - Batch size: {BATCH_SIZE} (optimized for throughput)")
    print("\nGenerated files:")
    print("  - training_metrics_enhanced.png")
    print("  - confusion_matrix_enhanced.png")
    print("  - per_class_accuracy_enhanced.png")
    print("  - best_model_enhanced.pth")
    print("="*80)


if __name__ == "__main__":
    total_start = time.time()
    main()
    total_end = time.time()
    print(f"\nTotal execution time: {total_end - total_start:.2f} seconds")