import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
# Set plotting style
plt.style.use('ggplot')
sns.set_palette("husl")

# Configuration
DATA_DIR = '/home/oussama/Desktop/MLA2/OIH_code/INVERTED'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15
IMG_SIZE = 128
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
            # Format: Label_Number.bmp (e.g., AiinI_1.bmp)
            try:
                # Split by underscore and take everything before the last underscore
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
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class OIH_CNN(nn.Module):
    def __init__(self, num_classes):
        super(OIH_CNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 -> 64
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
        )
        
        # After 3 pooling layers: 128 / 2^3 = 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    
    print("\n" + "="*60)
    print("Starting training...")
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
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ New best model saved (Val Acc: {val_acc:.2f}%)")
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
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
    axes[0].plot(epochs_range, train_losses, marker='o', linewidth=2, label='Train Loss')
    axes[0].plot(epochs_range, val_losses, marker='s', linewidth=2, label='Val Loss')
    axes[0].set_xlabel('Epochs', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy Plot
    axes[1].plot(epochs_range, train_accs, marker='o', linewidth=2, label='Train Acc')
    axes[1].plot(epochs_range, val_accs, marker='s', linewidth=2, label='Val Acc')
    axes[1].set_xlabel('Epochs', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Metrics plot saved to 'training_metrics.png'")
    plt.close()

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=0.5)
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to 'confusion_matrix.png'")
    plt.close()

def main():
    print("\n" + "="*60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print("="*60 + "\n")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load Dataset
    try:
        full_dataset = OIHDataset(DATA_DIR, transform=None)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # Train/Test Split with generator for reproducibility
    train_size = int((1 - TEST_SPLIT) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    # Apply different transforms
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    
    print(f"\nTraining on {len(train_dataset)} images")
    print(f"Validating on {len(test_dataset)} images")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}\n")
    
    # Model Setup
    num_classes = len(full_dataset.classes)
    model = OIH_CNN(num_classes).to(DEVICE)
    
    print("Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=EPOCHS
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    print("✓ Loaded best model for final evaluation\n")
    
    # Plotting
    plot_metrics(train_losses, train_accs, val_losses, val_accs)
    
    # Final Evaluation
    print("="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
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
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    total_start = time.time()
    # full training loop
    main()
    total_end = time.time()
    print(f"Total training time: {total_end - total_start:.2f} sec")
    