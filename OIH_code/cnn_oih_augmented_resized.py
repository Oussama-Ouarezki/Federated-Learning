import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import defaultdict
import time

# Set plotting style
plt.style.use('ggplot')
sns.set_palette("husl")

# Configuration - UPDATED PATH
DATA_DIR = '/home/oussama/Desktop/MLA2/OIH_code/RESIZED_32'  # Updated to 32x32 images folder
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15
WARMUP_EPOCHS = 0  # Linear warmup for first 3 epochs
IMG_SIZE = 32  # UPDATED: Changed from 128 to 32
TEST_SPLIT = 0.2
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class OIHDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = []
        
        # Scan directory
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")
            
        files = [f for f in os.listdir(root_dir) if f.lower().endswith('.bmp')]
        
        if len(files) == 0:
            raise ValueError(f"No BMP files found in {root_dir}")
        
        # Extract classes
        class_set = set()
        temp_data = []
        
        for filename in files:
            # Format: Label_Number.bmp
            try:
                parts = filename.rsplit('_', 1)
                if len(parts) == 2:
                    label = parts[0]
                    class_set.add(label)
                    temp_data.append((os.path.join(root_dir, filename), label))
                else:
                    print(f"Skipping file with unexpected format: {filename}")
            except Exception as e:
                print(f"Skipping file {filename}: {e}")
        
        if len(class_set) == 0:
            raise ValueError("No valid labels found in filenames")
                
        self.classes = sorted(list(class_set))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        for img_path, label in temp_data:
            self.image_paths.append(img_path)
            self.labels.append(self.class_to_idx[label])
        
        # Print class distribution
        class_counts = {}
        for label in self.labels:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        print(f"\nFound {len(self.image_paths)} images across {len(self.classes)} classes.")
        print(f"Classes: {self.classes}")
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

class OIH_CNN(nn.Module):
    def __init__(self, num_classes):
        super(OIH_CNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Grayscale input
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
        )
        
        # After 3 pooling layers: 32 / 2^3 = 4
        # Calculate the size: 128 channels * 4 * 4 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),  # UPDATED: Changed from 128*16*16 to 128*4*4
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, warmup_scheduler, epochs=10, warmup_epochs=3):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    
    print("\n" + "="*60)
    print("Starting training...")
    print(f"Warmup: {warmup_epochs} epochs | Cosine Annealing: {epochs - warmup_epochs} epochs")
    print("="*60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
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
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_32x32.pth')  # Updated filename
            print(f"✓ New best model saved (Val Acc: {val_acc:.2f}%)")
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] [{scheduler_name}] | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
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
    plt.savefig('training_metrics_32x32.png', dpi=300, bbox_inches='tight')  # Updated filename
    print("✓ Metrics plot saved to 'training_metrics_32x32.png'")
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
    plt.savefig('confusion_matrix_32x32.png', dpi=300, bbox_inches='tight')  # Updated filename
    print("✓ Confusion matrix saved to 'confusion_matrix_32x32.png'")
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
            
            for pred_class, count in sorted_preds:
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
    ax.set_title('Per-Class Accuracy (Tested on Original Non-Augmented Data)', fontsize=14, fontweight='bold')
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
    plt.savefig('per_class_accuracy_32x32.png', dpi=300, bbox_inches='tight')  # Updated filename
    print("✓ Per-class accuracy plot saved to 'per_class_accuracy_32x32.png'")
    plt.close()

def main():
    print("\n" + "="*80)
    print("ARABIC LETTER TRAINING - 32x32 IMAGES - NO HORIZONTAL FLIP")
    print("="*80)
    print(f"1. Loading 32x32 images from: {DATA_DIR}")
    print("2. Generate train/test indices")
    print("3. Create SEPARATE dataset instances with different transforms")
    print("4. Apply Subset to maintain proper separation")
    print("5. NO horizontal flip to preserve Arabic letter orientation")
    print("="*80 + "\n")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # STEP 1: Load original dataset (no augmentation yet)
    try:
        full_dataset = OIHDataset(DATA_DIR, transform=None)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # STEP 2: Get train/test indices BEFORE creating datasets
    indices = list(range(len(full_dataset)))
    train_size = int((1 - TEST_SPLIT) * len(indices))
    
    # Set seed and shuffle indices
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    print(f"\n{'='*60}")
    print(f"Dataset Split (BEFORE Augmentation):")
    print(f"  Training set: {len(train_indices)} images")
    print(f"  Test set: {len(test_indices)} images")
    print(f"{'='*60}\n")
    
    # STEP 3: Create SEPARATE dataset instances with different transforms
    
    # Training transform: AUGMENTATION WITHOUT HORIZONTAL FLIP (for Arabic letters)
    # Note: For 32x32 images, we need to adjust augmentation parameters
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # NO RandomHorizontalFlip - removed to preserve Arabic letter orientation
        transforms.RandomRotation(10),  # Reduced from ±15 to ±10 degrees for smaller images
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.05, 0.05),  # ±5% translation
            scale=(0.9, 1.1),        # Reduced scaling range for smaller images
            shear=5                  # Reduced shearing for smaller images
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Test transform: NO augmentation (original images only)
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create separate dataset instances
    train_dataset_full = OIHDataset(DATA_DIR, transform=train_transform)
    test_dataset_full = OIHDataset(DATA_DIR, transform=test_transform)
    
    # Apply the split using Subset
    train_dataset = Subset(train_dataset_full, train_indices)
    test_dataset = Subset(test_dataset_full, test_indices)
    
    print("✓ Created SEPARATE dataset instances for train and test")
    print("✓ Augmentation applied ONLY to training set")
    print("✓ NO horizontal flip - preserves Arabic letter orientation")
    print("✓ Test set uses original non-augmented 32x32 images")
    print("✓ Reduced augmentation intensity for 32x32 images\n")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    
    # Model Setup
    num_classes = len(full_dataset.classes)
    model = OIH_CNN(num_classes).to(DEVICE)
    
    print("Model Architecture (adapted for 32x32 images):")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Linear warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1,  # Start at 10% of base LR
        end_factor=1.0,    # End at 100% of base LR
        total_iters=WARMUP_EPOCHS
    )
    
    # Cosine annealing scheduler (applied after warmup)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS - WARMUP_EPOCHS,  # Cosine period
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Training
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer, 
        cosine_scheduler, warmup_scheduler, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_model_32x32.pth'))
    print("✓ Loaded best model for final evaluation\n")
    
    # Plotting training metrics
    plot_metrics(train_losses, train_accs, val_losses, val_accs)
    
    # Final Evaluation
    print("="*80)
    print("Final Evaluation on Test Set (Original Non-Augmented 32x32 Data)")
    print("="*80)
    
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
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, full_dataset.classes)
    
    # Per-class accuracy plot
    plot_per_class_accuracy(all_labels, all_preds, full_dataset.classes)
    
    # Detailed misclassification analysis
    analyze_misclassifications(all_labels, all_preds, full_dataset.classes)
    
    print("\n" + "="*80)
    print("Training completed successfully for 32x32 images!")
    print("="*80)
    print("ARABIC LETTER TRAINING Strategy for 32x32 images:")
    print("  - Input image size: 32x32 pixels")
    print("  - Created separate dataset instances for train/test")
    print("  - Training: Augmented data (reduced intensity for small images)")
    print("  - Training: NO horizontal flip (preserves Arabic orientation)")
    print("  - Testing: Original non-augmented 32x32 data")
    print("  - CNN architecture adapted for 32x32 input")
    print("\nGenerated files:")
    print("  - training_metrics_32x32.png (Loss & Accuracy plots)")
    print("  - confusion_matrix_32x32.png (Detailed confusion matrix)")
    print("  - per_class_accuracy_32x32.png (Per-class accuracy)")
    print("  - best_model_32x32.pth (Best model weights)")
    print("="*80)


if __name__ == "__main__":
    total_start = time.time()
    # full training loop
    main()
    total_end = time.time()
    print(f"Total training time: {total_end - total_start:.2f} sec")
    