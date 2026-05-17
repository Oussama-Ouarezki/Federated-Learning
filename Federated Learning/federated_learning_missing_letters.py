import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from collections import OrderedDict, defaultdict
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import warnings
import time

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set_palette("husl")

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Dataset Paths (Clients)
BASE_DIR = Path('/home/oussama/Desktop/MLA2/Federated Learning')
CLIENT_DIRS = {
    "0": BASE_DIR / "AHCD",
    "1": BASE_DIR / "HMBD",
    "2": BASE_DIR / "OIHCBD"
}

# Image Parameters
IMG_SIZE = 32
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LOCAL_EPOCHS = 5
NUM_ROUNDS = 50
RANDOM_SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Global variables to track metrics
global_metrics = {
    'round': [],
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': []
}

def get_params(model):
    """Extract model parameters as numpy arrays"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_params(model, parameters):
    """Set model parameters from numpy arrays"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

class UnifiedDataset(Dataset):
    """
    Dataset class for Arabic letters - works with both individual client folders
    and unified dataset structure
    """
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.image_paths = []
        self.labels = []
        
        valid_extensions = {'.bmp', '.png', '.jpg', '.jpeg'}
        
        if not self.root_dir.exists():
            print(f"Warning: Directory {self.root_dir} does not exist.")
            return

        # Scan files - handles format: Label_Number.ext (e.g., AiinI_100.png)
        for f in self.root_dir.iterdir():
            if f.suffix.lower() in valid_extensions:
                try:
                    stem = f.stem  # filename without extension
                    if '_' in stem:
                        # Split by last underscore to separate label from numbering
                        label_str = stem.rsplit('_', 1)[0]
                        if label_str in self.class_to_idx:
                            self.image_paths.append(str(f))
                            self.labels.append(self.class_to_idx[label_str])
                except Exception:
                    continue

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('L')  # Grayscale
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ArabicLetterCNN(nn.Module):
    """
    CNN model for Arabic Letter Recognition (32x32 grayscale)
    Same architecture as centralized version
    
    Architecture:
    - Block 1: Conv(1→32) → ReLU → BatchNorm → MaxPool → 16×16×32
    - Block 2: Conv(32→64) → ReLU → BatchNorm → MaxPool → 8×8×64
    - Block 3: Conv(64→128) → ReLU → BatchNorm → MaxPool → 4×4×128
    - Classifier: Flatten → Linear(2048→512) → ReLU → Dropout(0.5) → Linear(512→num_classes)
    """
    def __init__(self, num_classes):
        super(ArabicLetterCNN, self).__init__()
        
        # Block 1: 32x32 -> 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: 16x16 -> 8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: 8x8 -> 4x4
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),  # 2048 -> 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)  # 32x32x1 -> 16x16x32
        x = self.block2(x)  # 16x16x32 -> 8x8x64
        x = self.block3(x)  # 8x8x64 -> 4x4x128
        x = self.classifier(x)  # 4x4x128 -> num_classes
        return x

def train(model, train_loader, epochs=1):
    """Train the model for specified epochs"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        total_loss += epoch_loss
    
    avg_loss = total_loss / (len(train_loader) * epochs)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy

def test(model, test_loader):
    """Evaluate the model on test data"""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    correct, total, loss = 0, 0, 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = loss / len(test_loader) if len(test_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy

def evaluate_global_model(model, test_loaders, class_names):
    """Evaluate global model on all test data and return predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for test_loader in test_loaders:
            for images, labels in test_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
    
    return all_labels, all_preds

# --- FL Client ---
class ArabicClient(fl.client.NumPyClient):
    """Federated Learning Client for Arabic Letter Recognition"""
    
    def __init__(self, client_id, class_to_idx, num_classes):
        self.client_id = client_id
        self.class_to_idx = class_to_idx
        self.num_classes = num_classes
        
        # Load local data based on client_id
        data_dir = CLIENT_DIRS[client_id]
        print(f"\nClient {client_id}: Loading data from {data_dir.name}...")
        
        # Training transform WITH augmentation (NO horizontal flip for Arabic)
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(10),  # ±10 degrees
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),  # ±5% translation
                scale=(0.9, 1.1),  # 90-110% scaling
                shear=5  # Slight shearing
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Test transform WITHOUT augmentation
        test_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Create datasets with respective transforms
        train_dataset_full = UnifiedDataset(data_dir, class_to_idx, transform=train_transform)
        test_dataset_full = UnifiedDataset(data_dir, class_to_idx, transform=test_transform)
        
        # Train/Test split (80/20)
        total_size = len(train_dataset_full)
        if total_size == 0:
            raise ValueError(f"No valid images found in {data_dir}")
        
        indices = list(range(total_size))
        rng = np.random.RandomState(RANDOM_SEED)
        rng.shuffle(indices)
        
        train_size = int(0.8 * total_size)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # Create subsets
        self.train_set = Subset(train_dataset_full, train_indices)
        self.test_set = Subset(test_dataset_full, test_indices)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"  Client {client_id}: {len(self.train_set)} train samples, {len(self.test_set)} test samples")
        print(f"  Client {client_id}: Augmentation applied to training data only")
        
        # Initialize model
        self.model = ArabicLetterCNN(num_classes).to(DEVICE)

    def get_parameters(self, config):
        """Return current model parameters"""
        return get_params(self.model)

    def fit(self, parameters, config):
        """Train the model on local data"""
        set_params(self.model, parameters)
        
        # Train for specified local epochs
        train_loss, train_acc = train(self.model, self.train_loader, epochs=LOCAL_EPOCHS)
        
        print(f"  Client {self.client_id}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # FIXED: Return training metrics in the third element
        return get_params(self.model), len(self.train_set), {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc / 100.0)
        }

    def evaluate(self, parameters, config):
        """Evaluate the model on local test data"""
        set_params(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader)
        
        print(f"  Client {self.client_id}: Test Loss: {loss:.4f}, Test Acc: {100*accuracy:.2f}%")
        
        return float(loss), len(self.test_set), {"accuracy": float(accuracy)}

# --- Pre-scan to build global class map ---
def build_global_class_map():
    """Scan all client datasets to build a unified class mapping"""
    all_classes = set()
    valid_extensions = {'.bmp', '.png', '.jpg', '.jpeg'}
    
    print("\n" + "="*80)
    print("Scanning all client datasets to build global class map...")
    print("="*80)
    
    for client_id, dir_path in CLIENT_DIRS.items():
        if not dir_path.exists():
            print(f"Warning: Client {client_id} directory not found: {dir_path}")
            continue
        
        client_classes = set()
        for f in dir_path.iterdir():
            if f.suffix.lower() in valid_extensions:
                stem = f.stem
                if '_' in stem:
                    label = stem.rsplit('_', 1)[0]
                    client_classes.add(label)
                    all_classes.add(label)
        
        print(f"Client {client_id} ({dir_path.name}): {len(client_classes)} unique classes")
        print(f"  Classes: {sorted(client_classes)}")
    
    sorted_classes = sorted(list(all_classes))
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}
    
    print(f"\nTotal unique classes across all clients: {len(sorted_classes)}")
    print(f"Classes: {sorted_classes}")
    print("="*80 + "\n")
    
    return class_to_idx, len(sorted_classes)

def plot_federated_metrics(metrics):
    """Plot training metrics over federated rounds"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = metrics['round']
    
    # Loss Plot
    axes[0].plot(rounds, metrics['train_loss'], marker='o', linewidth=2, 
                label='Train Loss (Avg)', markersize=6, color='#1f77b4')
    axes[0].plot(rounds, metrics['test_loss'], marker='s', linewidth=2, 
                label='Test Loss (Avg)', markersize=6, color='#ff7f0e')
    axes[0].set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Federated Learning - Loss Over Rounds', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy Plot
    axes[1].plot(rounds, metrics['train_acc'], marker='o', linewidth=2, 
                label='Train Acc (Avg)', markersize=6, color='#2ca02c')
    axes[1].plot(rounds, metrics['test_acc'], marker='s', linewidth=2, 
                label='Test Acc (Avg)', markersize=6, color='#d62728')
    axes[1].set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Federated Learning - Accuracy Over Rounds', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('federated_training_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Training metrics plot saved to 'federated_training_metrics.png'")
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
    ax.set_title('Federated Learning - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('federated_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to 'federated_confusion_matrix.png'")
    plt.close()

def plot_per_class_accuracy(all_labels, all_preds, class_names):
    """Plot per-class accuracy bar chart"""
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
    ax.set_title('Federated Learning - Per-Class Accuracy', fontsize=14, fontweight='bold')
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
    plt.savefig('federated_per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print("✓ Per-class accuracy plot saved to 'federated_per_class_accuracy.png'")
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

# FIXED: Custom strategy to properly track metrics and save global model
class MetricsStrategy(fl.server.strategy.FedAvg):
    def __init__(self, global_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = global_model
        self.num_classes = global_model.classifier[-1].out_features
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and track training metrics"""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # FIXED: Update the global model with aggregated parameters
            params_list = parameters_to_ndarrays(aggregated_parameters)
            set_params(self.global_model, params_list)
            
            # FIXED: Calculate weighted average of training metrics
            if results:
                total_examples = sum([r.num_examples for _, r in results])
                
                # Extract training metrics from client results
                train_losses = [r.metrics.get("train_loss", 0.0) * r.num_examples 
                               for _, r in results if "train_loss" in r.metrics]
                train_accs = [r.metrics.get("train_accuracy", 0.0) * r.num_examples 
                             for _, r in results if "train_accuracy" in r.metrics]
                
                if train_losses and total_examples > 0:
                    avg_train_loss = sum(train_losses) / total_examples
                    avg_train_acc = sum(train_accs) / total_examples
                    
                    global_metrics['train_loss'].append(avg_train_loss)
                    global_metrics['train_acc'].append(avg_train_acc * 100)
                    
                    print(f"\n[Round {server_round}] Aggregated Train Loss: {avg_train_loss:.4f}, "
                          f"Train Accuracy: {avg_train_acc*100:.2f}%")
            
            # Save checkpoint every 10 rounds
            if server_round % 10 == 0:
                torch.save(self.global_model.state_dict(), 
                          f'checkpoint_round_{server_round}.pth')
                print(f"✓ Checkpoint saved: checkpoint_round_{server_round}.pth")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results and track test metrics"""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        if aggregated_loss is not None:
            # Calculate weighted average of accuracies
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]
            avg_accuracy = sum(accuracies) / sum(examples) if sum(examples) > 0 else 0
            
            # Track metrics
            global_metrics['round'].append(server_round)
            global_metrics['test_loss'].append(aggregated_loss)
            global_metrics['test_acc'].append(avg_accuracy * 100)
            
            print(f"[Round {server_round}] Aggregated Test Loss: {aggregated_loss:.4f}, "
                  f"Test Accuracy: {avg_accuracy*100:.2f}%")
        
        return aggregated_loss, aggregated_metrics

def comprehensive_model_evaluation(model, class_to_idx, class_names):
    """
    Comprehensive evaluation on ALL data (train + test combined) from all clients
    Shows how well the model learned each letter across the entire dataset
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION - FINAL INFERENCE ON ALL DATA")
    print("="*80)
    print("Testing the trained global model on ALL available data...")
    print("(This combines both training and test sets from all clients)")
    print("="*80 + "\n")
    
    model.eval()
    
    # Collect predictions and labels from ALL data (no train/test split)
    all_preds = []
    all_labels = []
    all_probabilities = []
    
    # Per-client statistics
    client_stats = {}
    
    # Test transform (same as evaluation - no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    for client_id, data_dir in CLIENT_DIRS.items():
        print(f"\nProcessing Client {client_id} ({data_dir.name})...")
        
        # Load ALL data from this client (no split)
        full_dataset = UnifiedDataset(data_dir, class_to_idx, transform=test_transform)
        
        if len(full_dataset) == 0:
            print(f"  Warning: No data found for client {client_id}")
            continue
        
        full_loader = DataLoader(
            full_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        client_preds = []
        client_labels = []
        client_correct = 0
        client_total = 0
        
        with torch.no_grad():
            for images, labels in full_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                
                # Get probabilities
                probs = torch.softmax(outputs, dim=1)
                all_probabilities.extend(probs.cpu().numpy())
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                client_preds.extend(predicted.cpu().numpy())
                client_labels.extend(labels.numpy())
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                
                client_total += labels.size(0)
                client_correct += (predicted.cpu() == labels).sum().item()
        
        client_accuracy = 100.0 * client_correct / client_total if client_total > 0 else 0.0
        
        # Get unique classes in this client
        client_classes = sorted(set(client_labels))
        client_class_names = [class_names[i] for i in client_classes]
        
        client_stats[client_id] = {
            'name': data_dir.name,
            'total_samples': client_total,
            'correct': client_correct,
            'accuracy': client_accuracy,
            'classes': client_class_names,
            'num_classes': len(client_class_names)
        }
        
        print(f"  Samples: {client_total}")
        print(f"  Classes: {len(client_class_names)} ({', '.join(client_class_names)})")
        print(f"  Accuracy: {client_accuracy:.2f}%")
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probabilities = np.array(all_probabilities)
    
    # Overall statistics
    total_samples = len(all_labels)
    total_correct = np.sum(all_labels == all_preds)
    overall_accuracy = 100.0 * total_correct / total_samples
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total samples evaluated: {total_samples}")
    print(f"Total correct predictions: {total_correct}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    print("="*80)
    
    return all_labels, all_preds, all_probabilities, client_stats

def plot_detailed_per_letter_accuracy(all_labels, all_preds, class_names, client_stats):
    """
    Create a detailed per-letter accuracy visualization showing:
    1. Accuracy per letter
    2. Sample count per letter
    3. Which client each letter belongs to
    """
    print("\n" + "="*80)
    print("GENERATING DETAILED PER-LETTER ACCURACY ANALYSIS")
    print("="*80)
    
    # Calculate per-class metrics
    per_class_metrics = []
    
    for i, class_name in enumerate(class_names):
        # Get all samples for this class
        class_mask = all_labels == i
        class_true = all_labels[class_mask]
        class_pred = all_preds[class_mask]
        
        if len(class_true) > 0:
            accuracy = accuracy_score(class_true, class_pred) * 100
            correct = np.sum(class_true == class_pred)
            total = len(class_true)
        else:
            accuracy = 0.0
            correct = 0
            total = 0
        
        # Determine which client this class belongs to
        client_name = "Unknown"
        for cid, stats in client_stats.items():
            if class_name in stats['classes']:
                client_name = stats['name']
                break
        
        per_class_metrics.append({
            'class': class_name,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'client': client_name
        })
    
    # Sort by client and then by class name
    client_order = {'AHCD': 0, 'HMBD': 1, 'OIHCBD': 2}
    per_class_metrics.sort(key=lambda x: (client_order.get(x['client'], 999), x['class']))
    
    # Create detailed bar plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(16, len(class_names) * 0.7), 12))
    
    classes = [m['class'] for m in per_class_metrics]
    accuracies = [m['accuracy'] for m in per_class_metrics]
    totals = [m['total'] for m in per_class_metrics]
    clients = [m['client'] for m in per_class_metrics]
    
    # Plot 1: Accuracy per letter with client colors
    colors = []
    client_colors = {'AHCD': '#FF6B6B', 'HMBD': '#4ECDC4', 'OIHCBD': '#45B7D1'}
    for client in clients:
        colors.append(client_colors.get(client, '#95A5A6'))
    
    bars1 = ax1.bar(range(len(classes)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (acc, total, correct) in enumerate(zip(accuracies, totals, [m['correct'] for m in per_class_metrics])):
        if total > 0:
            ax1.text(i, acc + 1, f'{acc:.1f}%\n{correct}/{total}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_xlabel('Letter (Arabic Character)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Per-Letter Accuracy - Comprehensive Evaluation on All Data', 
                  fontsize=15, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    ax1.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    
    # Add legend for clients
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='AHCD', edgecolor='black'),
        Patch(facecolor='#4ECDC4', label='HMBD', edgecolor='black'),
        Patch(facecolor='#45B7D1', label='OIHCBD', edgecolor='black'),
        Patch(facecolor='none', edgecolor='green', label='≥90% (Excellent)', linestyle='--'),
        Patch(facecolor='none', edgecolor='orange', label='70-89% (Good)', linestyle='--')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10, ncol=2)
    
    # Plot 2: Sample count per letter
    bars2 = ax2.bar(range(len(classes)), totals, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, total in enumerate(totals):
        ax2.text(i, total + max(totals)*0.01, f'{total}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Letter (Arabic Character)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
    ax2.set_title('Sample Distribution per Letter', fontsize=15, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('final_per_letter_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Detailed per-letter analysis saved to 'final_per_letter_detailed_analysis.png'")
    plt.close()
    
    return per_class_metrics

def print_per_letter_table(per_class_metrics):
    """Print a detailed table of per-letter accuracy"""
    print("\n" + "="*80)
    print("DETAILED PER-LETTER ACCURACY TABLE")
    print("="*80)
    
    # Group by client
    clients = {}
    for metric in per_class_metrics:
        client = metric['client']
        if client not in clients:
            clients[client] = []
        clients[client].append(metric)
    
    for client_name in ['AHCD', 'HMBD', 'OIHCBD']:
        if client_name not in clients:
            continue
        
        print(f"\n{'─'*80}")
        print(f"  {client_name}")
        print(f"{'─'*80}")
        print(f"{'Letter':<15} {'Accuracy':<12} {'Correct':<10} {'Total':<10} {'Status'}")
        print(f"{'─'*80}")
        
        client_metrics = clients[client_name]
        client_total_correct = sum(m['correct'] for m in client_metrics)
        client_total_samples = sum(m['total'] for m in client_metrics)
        
        for metric in client_metrics:
            status = "✓ Excellent" if metric['accuracy'] >= 90 else \
                     "○ Good" if metric['accuracy'] >= 70 else \
                     "✗ Needs Work"
            
            print(f"{metric['class']:<15} {metric['accuracy']:>6.2f}%      "
                  f"{metric['correct']:>4}/{metric['total']:<5}  {metric['total']:<10} {status}")
        
        client_avg = 100.0 * client_total_correct / client_total_samples if client_total_samples > 0 else 0
        print(f"{'─'*80}")
        print(f"{'CLIENT AVERAGE':<15} {client_avg:>6.2f}%      "
              f"{client_total_correct:>4}/{client_total_samples:<5}")
        print(f"{'─'*80}")
    
    # Overall summary
    total_correct = sum(m['correct'] for m in per_class_metrics)
    total_samples = sum(m['total'] for m in per_class_metrics)
    overall_avg = 100.0 * total_correct / total_samples if total_samples > 0 else 0
    
    print(f"\n{'═'*80}")
    print(f"  OVERALL SUMMARY")
    print(f"{'═'*80}")
    print(f"{'TOTAL ACCURACY':<15} {overall_avg:>6.2f}%      {total_correct:>4}/{total_samples:<5}")
    
    # Performance categories
    excellent = sum(1 for m in per_class_metrics if m['accuracy'] >= 90)
    good = sum(1 for m in per_class_metrics if 70 <= m['accuracy'] < 90)
    needs_work = sum(1 for m in per_class_metrics if m['accuracy'] < 70)
    
    print(f"\nPerformance Distribution:")
    print(f"  ✓ Excellent (≥90%):    {excellent}/{len(per_class_metrics)} letters")
    print(f"  ○ Good (70-89%):       {good}/{len(per_class_metrics)} letters")
    print(f"  ✗ Needs Work (<70%):   {needs_work}/{len(per_class_metrics)} letters")
    print(f"{'═'*80}\n")

def analyze_worst_performing_letters(all_labels, all_preds, all_probabilities, class_names):
    """Identify and analyze the worst performing letters"""
    print("\n" + "="*80)
    print("WORST PERFORMING LETTERS ANALYSIS")
    print("="*80)
    
    per_class_acc = []
    for i, class_name in enumerate(class_names):
        class_mask = all_labels == i
        class_true = all_labels[class_mask]
        class_pred = all_preds[class_mask]
        
        if len(class_true) > 0:
            accuracy = accuracy_score(class_true, class_pred) * 100
            per_class_acc.append((class_name, accuracy, len(class_true), i))
    
    # Sort by accuracy
    per_class_acc.sort(key=lambda x: x[1])
    
    print("\n🔴 TOP 5 WORST PERFORMING LETTERS:\n")
    for rank, (letter, acc, count, idx) in enumerate(per_class_acc[:5], 1):
        print(f"{rank}. {letter:<15} Accuracy: {acc:>6.2f}%  (n={count})")
        
        # Show what it's being confused with
        class_mask = all_labels == idx
        confused_preds = all_preds[class_mask]
        
        # Count misclassifications
        misclass = {}
        for pred in confused_preds:
            if pred != idx:
                misclass[pred] = misclass.get(pred, 0) + 1
        
        if misclass:
            # Sort by frequency
            sorted_misclass = sorted(misclass.items(), key=lambda x: x[1], reverse=True)
            print(f"   Most confused with:")
            for confused_idx, count_confused in sorted_misclass[:3]:
                print(f"      → {class_names[confused_idx]}: {count_confused} times")
        print()
    
    print("\n🟢 TOP 5 BEST PERFORMING LETTERS:\n")
    for rank, (letter, acc, count, idx) in enumerate(per_class_acc[-5:][::-1], 1):
        print(f"{rank}. {letter:<15} Accuracy: {acc:>6.2f}%  (n={count})")
    
    print("\n" + "="*80)

def save_final_results_summary(overall_accuracy, per_class_metrics, client_stats):
    """Save a comprehensive text summary of results"""
    with open('final_evaluation_summary.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FEDERATED LEARNING - FINAL EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
        f.write(f"Total Classes: {len(per_class_metrics)}\n")
        f.write(f"Total Samples: {sum(m['total'] for m in per_class_metrics)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("PER-CLIENT PERFORMANCE\n")
        f.write("="*80 + "\n\n")
        
        for cid, stats in client_stats.items():
            f.write(f"{stats['name']}:\n")
            f.write(f"  Samples: {stats['total_samples']}\n")
            f.write(f"  Classes: {stats['num_classes']}\n")
            f.write(f"  Accuracy: {stats['accuracy']:.2f}%\n")
            f.write(f"  Letters: {', '.join(stats['classes'])}\n\n")
        
        f.write("="*80 + "\n")
        f.write("PER-LETTER DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Letter':<15} {'Client':<12} {'Accuracy':<12} {'Correct/Total':<15}\n")
        f.write("-"*80 + "\n")
        
        for metric in per_class_metrics:
            f.write(f"{metric['class']:<15} {metric['client']:<12} "
                   f"{metric['accuracy']:>6.2f}%      "
                   f"{metric['correct']:>4}/{metric['total']:<10}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print("✓ Detailed summary saved to 'final_evaluation_summary.txt'")

def main():
    start_time = time.time()
    
    print("\n" + "="*80)
    print("FEDERATED LEARNING - ARABIC LETTER RECOGNITION (FIXED VERSION)")
    print("="*80)
    print(f"Framework: Flower (Federated Learning)")
    print(f"Number of clients: {len(CLIENT_DIRS)}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE} (grayscale)")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Local epochs per round: {LOCAL_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of rounds: {NUM_ROUNDS}")
    print(f"Device: {DEVICE}")
    print("="*80)
    
    # 1. Global Setup - Build unified class mapping
    global_class_to_idx, num_classes = build_global_class_map()
    
    if num_classes == 0:
        print("Error: No classes found. Please check dataset paths.")
        return
    
    # Get class names for visualization
    class_names = sorted(global_class_to_idx.keys())
    
    # FIXED: Create global model BEFORE strategy
    global_model = ArabicLetterCNN(num_classes).to(DEVICE)
    print(f"\n✓ Global model initialized with {num_classes} classes")

    # 2. Define Client Function
    def client_fn(cid: str):
        """Create a client instance"""
        return ArabicClient(cid, global_class_to_idx, num_classes)

    # 3. FIXED: Define Strategy with global model reference
    strategy = MetricsStrategy(
        global_model=global_model,
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=3,  # Minimum clients for training
        min_evaluate_clients=3,  # Minimum clients for evaluation
        min_available_clients=3,  # Minimum available clients to start
    )

    # 4. Start Simulation
    print("\n" + "="*80)
    print("Starting Federated Learning Simulation...")
    print("="*80)
    print(f"Clients: {[d.name for d in CLIENT_DIRS.values()]}")
    print("Strategy: FedAvg (Federated Averaging) with Metrics Tracking")
    print("="*80 + "\n")
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.2 if torch.cuda.is_available() else 0  # Reduced from 0.25
        }
    )
    
    print("\n" + "="*80)
    print("Federated Learning Completed!")
    print("="*80)
    
    # 5. Plot training metrics
    if len(global_metrics['round']) > 0:
        plot_federated_metrics(global_metrics)
    
    # 6. COMPREHENSIVE FINAL EVALUATION - Test on ALL data
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE FINAL EVALUATION")
    print("="*80)
    
    # Perform comprehensive evaluation on all available data
    all_labels, all_preds, all_probabilities, client_stats = comprehensive_model_evaluation(
        global_model, global_class_to_idx, class_names
    )
    
    # 7. Generate detailed per-letter accuracy analysis
    per_class_metrics = plot_detailed_per_letter_accuracy(
        all_labels, all_preds, class_names, client_stats
    )
    
    # 8. Print detailed per-letter table
    print_per_letter_table(per_class_metrics)
    
    # 9. Analyze worst and best performing letters
    analyze_worst_performing_letters(all_labels, all_preds, all_probabilities, class_names)
    
    # 10. Generate classification report (on test data only for standard metrics)
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT (on held-out test sets)")
    print("="*80)
    
    # Create test loaders for standard evaluation
    test_loaders = []
    for cid in ["0", "1", "2"]:
        client = ArabicClient(cid, global_class_to_idx, num_classes)
        test_loaders.append(client.test_loader)
    
    test_labels, test_preds = evaluate_global_model(global_model, test_loaders, class_names)
    
    print("\nClassification Report (Test Set):")
    print(classification_report(test_labels, test_preds, target_names=class_names, digits=4))
    
    # 11. Generate confusion matrix (on test data)
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, class_names)
    
    # 12. Generate per-class accuracy plot (on test data)
    plot_per_class_accuracy(test_labels, test_preds, class_names)
    
    # 13. Analyze misclassifications (on test data)
    analyze_misclassifications(test_labels, test_preds, class_names)
    
    # 14. Save comprehensive summary
    overall_accuracy = 100.0 * np.sum(all_labels == all_preds) / len(all_labels)
    save_final_results_summary(overall_accuracy, per_class_metrics, client_stats)
    
    # 15. Save final model
    torch.save(global_model.state_dict(), 'federated_global_model_FIXED.pth')
    print("\n✓ Final global model saved to 'federated_global_model_FIXED.pth'")
    
    end_time = time.time()
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Training Strategy: Federated Learning (FedAvg)")
    print(f"Number of clients: 3 (AHCD, HMBD, OIHCBD)")
    print(f"Federated rounds: {NUM_ROUNDS}")
    print(f"Local epochs per round: {LOCAL_EPOCHS}")
    print(f"Total effective epochs per client: {NUM_ROUNDS * LOCAL_EPOCHS}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE} (grayscale)")
    print(f"Architecture: 3 conv blocks (32→64→128 channels)")
    print(f"Augmentation: Rotation, affine, brightness (NO horizontal flip)")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    print("\n" + "─"*80)
    print("FINAL MODEL PERFORMANCE:")
    print("─"*80)
    print(f"Overall Accuracy (all data): {overall_accuracy:.2f}%")
    print(f"Test Set Accuracy: {100.0 * np.sum(test_labels == test_preds) / len(test_labels):.2f}%")
    print(f"Total letters evaluated: {len(class_names)}")
    
    # Performance breakdown
    excellent = sum(1 for m in per_class_metrics if m['accuracy'] >= 90)
    good = sum(1 for m in per_class_metrics if 70 <= m['accuracy'] < 90)
    needs_work = sum(1 for m in per_class_metrics if m['accuracy'] < 70)
    
    print(f"\nLetter Performance Distribution:")
    print(f"  ✓ Excellent (≥90%):    {excellent}/{len(class_names)} letters ({100*excellent/len(class_names):.1f}%)")
    print(f"  ○ Good (70-89%):       {good}/{len(class_names)} letters ({100*good/len(class_names):.1f}%)")
    print(f"  ✗ Needs Work (<70%):   {needs_work}/{len(class_names)} letters ({100*needs_work/len(class_names):.1f}%)")
    
    print("\n" + "─"*80)
    print("GENERATED FILES:")
    print("─"*80)
    print("Training Visualizations:")
    print("  - federated_training_metrics.png (Loss & Accuracy over rounds)")
    print("\nTest Set Evaluation:")
    print("  - federated_confusion_matrix.png (Confusion matrix)")
    print("  - federated_per_class_accuracy.png (Per-class accuracy)")
    print("\nComprehensive Final Evaluation (ALL data):")
    print("  - final_per_letter_detailed_analysis.png (Detailed per-letter breakdown)")
    print("  - final_evaluation_summary.txt (Complete text summary)")
    print("\nModel Checkpoints:")
    print("  - federated_global_model_FIXED.pth (Final trained global model)")
    print("  - checkpoint_round_X.pth (Checkpoints every 10 rounds)")
    
    print("\n" + "="*80)
    print("✅ FIXES APPLIED IN THIS VERSION:")
    print("="*80)
    print("  1. Global model is now properly updated during training")
    print("  2. Training metrics are correctly tracked from client results")
    print("  3. Final evaluation uses the TRAINED model (not random)")
    print("  4. Checkpoints saved every 10 rounds")
    print("  5. GPU allocation reduced to 0.2 per client (was 0.25)")
    print("  6. ⭐ NEW: Comprehensive inference on ALL data with per-letter analysis")
    print("  7. ⭐ NEW: Detailed per-letter accuracy table grouped by client")
    print("  8. ⭐ NEW: Best/worst performing letters analysis")
    print("  9. ⭐ NEW: Complete evaluation summary saved to text file")
    print("="*80)
    
    print("\n🎯 KEY INSIGHTS:")
    print("-"*80)
    print("The model was evaluated on:")
    print(f"  • {sum(stats['total_samples'] for stats in client_stats.values())} total samples (ALL data)")
    print(f"  • {len(test_labels)} test samples (held-out evaluation)")
    print(f"  • {len(class_names)} unique Arabic letters")
    print(f"  • 3 different data sources (non-IID federated setup)")
    print("\nThis comprehensive evaluation shows exactly how well the federated model")
    print("learned each individual Arabic letter across all available data!")
    print("="*80)

if __name__ == "__main__":
    total_start = time.time()
    main()
    total_end = time.time()
    print(f"\n⏱️  Total training time: {total_end - total_start:.2f} seconds")