"""
CNN Models for Arabic Letter Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for Arabic Letter Classification
    """
    
    def __init__(self, num_classes=28, input_channels=3):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 3 pooling layers: 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 8 * 8)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ImprovedCNN(nn.Module):
    """
    Improved CNN with deeper architecture and residual connections
    """
    
    def __init__(self, num_classes=28, input_channels=3):
        super(ImprovedCNN, self).__init__()
        
        # Initial conv
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Conv blocks
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Conv blocks with pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for deeper CNN"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetLike(nn.Module):
    """
    ResNet-like architecture for Arabic letter recognition
    """
    
    def __init__(self, num_classes=28, input_channels=3):
        super(ResNetLike, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def get_model(model_name='simple', num_classes=28, input_channels=3):
    """
    Factory function to get model by name
    
    Args:
        model_name: 'simple', 'improved', or 'resnet'
        num_classes: number of output classes
        input_channels: number of input channels (3 for RGB)
    """
    models = {
        'simple': SimpleCNN,
        'improved': ImprovedCNN,
        'resnet': ResNetLike
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
    
    model = models[model_name](num_classes=num_classes, input_channels=input_channels)
    return model


# Test models
if __name__ == "__main__":
    # Test all models
    models_to_test = ['simple', 'improved', 'resnet']
    batch_size = 4
    num_classes = 28
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name} model:")
        model = get_model(model_name, num_classes=num_classes)
        
        # Create dummy input
        x = torch.randn(batch_size, 3, 64, 64)
        
        # Forward pass
        output = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
