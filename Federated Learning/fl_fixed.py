import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import flwr as fl
from collections import OrderedDict
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

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
RANDOM_SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
        
        return get_params(self.model), len(self.train_set), {}

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
    
    sorted_classes = sorted(list(all_classes))
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}
    
    print(f"\nTotal unique classes across all clients: {len(sorted_classes)}")
    print(f"Classes: {sorted_classes}")
    print("="*80 + "\n")
    
    return class_to_idx, len(sorted_classes)

def main():
    print("\n" + "="*80)
    print("FEDERATED LEARNING - ARABIC LETTER RECOGNITION")
    print("="*80)
    print(f"Framework: Flower (Federated Learning)")
    print(f"Number of clients: {len(CLIENT_DIRS)}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE} (grayscale)")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Local epochs per round: {LOCAL_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Device: {DEVICE}")
    print("="*80)
    
    # 1. Global Setup - Build unified class mapping
    global_class_to_idx, num_classes = build_global_class_map()
    
    if num_classes == 0:
        print("Error: No classes found. Please check dataset paths.")
        return

    # 2. Define Client Function
    def client_fn(cid: str):
        """Create a client instance"""
        return ArabicClient(cid, global_class_to_idx, num_classes)

    # 3. Define Strategy (FedAvg with improved settings)
    strategy = fl.server.strategy.FedAvg(
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
    print(f"Clients: {list(CLIENT_DIRS.values())}")
    print("Strategy: FedAvg (Federated Averaging)")
    print("="*80 + "\n")
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=50),  # 50 federated rounds
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.25 if torch.cuda.is_available() else 0
        }
    )
    
    print("\n" + "="*80)
    print("Federated Learning Completed!")
    print("="*80)

if __name__ == "__main__":
    main()