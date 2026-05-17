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

plt.style.use('ggplot')
sns.set_palette("husl")

# ================= CONFIG =================
DATA_DIR = '/home/oussama/Desktop/MLA2/centralized learning/unified_dataset'
METADATA_CSV = os.path.join(DATA_DIR, 'dataset_metadata.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
WARMUP_EPOCHS = 3
IMG_SIZE = 32
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
RANDOM_SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ================= DATASET =================
class UnifiedArabicDataset(Dataset):
    def __init__(self, metadata_csv, images_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform

        self.metadata = pd.read_csv(metadata_csv)

        # 🔴 REMOVE HamzaI
        self.metadata = self.metadata[self.metadata['label'] != 'HamzaI'].reset_index(drop=True)

        self.classes = sorted(self.metadata['label'].unique().tolist())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        for _, row in self.metadata.iterrows():
            img_path = self.images_dir / row['filename']
            if img_path.exists():
                self.image_paths.append(str(img_path))
                self.labels.append(self.class_to_idx[row['label']])

        print(f"\nLoaded {len(self.image_paths)} images")
        print(f"Classes ({len(self.classes)}): {self.classes}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ================= MODEL =================
class ArabicLetterCNN(nn.Module):
    """
    Less downsampling + deeper network + GAP
    """
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 (32x32)
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 32 → 16

            # Block 2 (16x16)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 16 → 8

            # Block 3 (NO pooling)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ================= TRAIN / EVAL =================
def evaluate_model(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / len(loader), 100 * correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, warmup):
    best_acc, patience, wait = 0, 10, 0
    for epoch in range(EPOCHS):
        model.train()
        correct, total, loss_sum = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

        if epoch < WARMUP_EPOCHS:
            warmup.step()
        else:
            scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), 'best_model_unified.pth')
        else:
            wait += 1

        print(f"Epoch {epoch+1:02d} | Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}%")

        if wait >= patience:
            print("Early stopping triggered")
            break

# ================= MAIN =================
def main():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.05,0.05), scale=(0.9,1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    base_ds = UnifiedArabicDataset(METADATA_CSV, IMAGES_DIR)
    indices = np.arange(len(base_ds))
    np.random.shuffle(indices)

    n_test = int(TEST_SPLIT * len(indices))
    n_val = int(VAL_SPLIT * len(indices))

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test+n_val]
    train_idx = indices[n_test+n_val:]

    train_ds = Subset(UnifiedArabicDataset(METADATA_CSV, IMAGES_DIR, train_tf), train_idx)
    val_ds   = Subset(UnifiedArabicDataset(METADATA_CSV, IMAGES_DIR, eval_tf), val_idx)
    test_ds  = Subset(UnifiedArabicDataset(METADATA_CSV, IMAGES_DIR, eval_tf), test_idx)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, BATCH_SIZE)
    test_loader  = DataLoader(test_ds, BATCH_SIZE)

    model = ArabicLetterCNN(len(base_ds.classes)).to(DEVICE)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    warmup = optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, WARMUP_EPOCHS)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS - WARMUP_EPOCHS)

    train_model(model, train_loader, val_loader, criterion, optimizer, cosine, warmup)

    model.load_state_dict(torch.load('best_model_unified.pth'))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(DEVICE))
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())

    print(classification_report(y_true, y_pred, target_names=base_ds.classes))

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nDone in {time.time() - start:.2f}s")
