"""
Dataset Analysis and Visualization Utilities
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from PIL import Image
import random


def analyze_dataset(metadata_path, save_dir='analysis'):
    """
    Analyze the unified dataset and create visualizations
    
    Args:
        metadata_path: path to dataset_metadata.csv
        save_dir: directory to save analysis plots
    """
    # Load metadata
    df = pd.read_csv(metadata_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"\nTotal images: {len(df)}")
    print(f"Number of classes: {df['label'].nunique()}")
    print(f"Number of datasets: {df['dataset'].nunique()}")
    
    # Distribution by dataset
    print("\n" + "-"*60)
    print("Distribution by Dataset:")
    print("-"*60)
    dataset_counts = df['dataset'].value_counts()
    for dataset, count in dataset_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {dataset}: {count} ({percentage:.1f}%)")
    
    # Distribution by class
    print("\n" + "-"*60)
    print("Distribution by Class:")
    print("-"*60)
    class_counts = df['label'].value_counts().sort_index()
    for label, count in class_counts.items():
        print(f"  {label}: {count}")
    
    # Check for class imbalance
    min_count = class_counts.min()
    max_count = class_counts.max()
    imbalance_ratio = max_count / min_count
    print(f"\n  Min samples per class: {min_count}")
    print(f"  Max samples per class: {max_count}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 2.0:
        print("  ⚠️  Warning: Significant class imbalance detected!")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution by dataset (pie chart)
    axes[0, 0].pie(dataset_counts.values, labels=dataset_counts.index, 
                    autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Distribution by Dataset', fontsize=14, fontweight='bold')
    
    # 2. Distribution by class (bar chart)
    class_counts.plot(kind='bar', ax=axes[0, 1], color='skyblue')
    axes[0, 1].set_title('Distribution by Class', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Class Label')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Heatmap: Dataset vs Class
    pivot_table = df.pivot_table(
        index='dataset', 
        columns='label', 
        aggfunc='size', 
        fill_value=0
    )
    sns.heatmap(pivot_table, annot=False, cmap='YlOrRd', ax=axes[1, 0], 
                cbar_kws={'label': 'Count'})
    axes[1, 0].set_title('Dataset vs Class Heatmap', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Class Label')
    axes[1, 0].set_ylabel('Dataset')
    
    # 4. Box plot: Samples per class across datasets
    dataset_class_counts = df.groupby(['dataset', 'label']).size().reset_index(name='count')
    dataset_class_counts.boxplot(column='count', by='dataset', ax=axes[1, 1])
    axes[1, 1].set_title('Samples per Class Distribution by Dataset', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Dataset')
    axes[1, 1].set_ylabel('Samples per Class')
    plt.suptitle('')  # Remove automatic title
    
    plt.tight_layout()
    plt.savefig(save_dir / 'dataset_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Analysis plot saved to {save_dir / 'dataset_analysis.png'}")
    plt.close()
    
    return df


def visualize_samples(metadata_path, images_dir, num_samples=20, save_path='sample_grid.png'):
    """
    Visualize random samples from the dataset
    
    Args:
        metadata_path: path to dataset_metadata.csv
        images_dir: path to images directory
        num_samples: number of samples to show
        save_path: path to save the visualization
    """
    df = pd.read_csv(metadata_path)
    images_dir = Path(images_dir)
    
    # Randomly sample images
    sample_df = df.sample(n=min(num_samples, len(df)))
    
    # Calculate grid size
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        if idx >= num_samples:
            break
            
        img_path = images_dir / row['filename']
        
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"{row['label']}\n({row['dataset']})", 
                               fontsize=10)
            axes[idx].axis('off')
        except Exception as e:
            axes[idx].text(0.5, 0.5, 'Error loading', ha='center', va='center')
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(sample_df), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Random Dataset Samples', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Sample visualization saved to {save_path}")
    plt.close()


def visualize_class_samples(metadata_path, images_dir, num_classes=10, 
                           samples_per_class=5, save_path='class_samples.png'):
    """
    Visualize samples for each class
    
    Args:
        metadata_path: path to dataset_metadata.csv
        images_dir: path to images directory
        num_classes: number of classes to visualize
        samples_per_class: number of samples per class
        save_path: path to save the visualization
    """
    df = pd.read_csv(metadata_path)
    images_dir = Path(images_dir)
    
    # Get random classes
    classes = df['label'].unique()
    selected_classes = np.random.choice(classes, size=min(num_classes, len(classes)), 
                                       replace=False)
    
    fig, axes = plt.subplots(num_classes, samples_per_class, 
                            figsize=(samples_per_class*2, num_classes*2))
    
    for class_idx, label in enumerate(selected_classes):
        # Get samples for this class
        class_df = df[df['label'] == label].sample(n=min(samples_per_class, len(df[df['label'] == label])))
        
        for sample_idx, (_, row) in enumerate(class_df.iterrows()):
            if sample_idx >= samples_per_class:
                break
                
            img_path = images_dir / row['filename']
            
            try:
                img = Image.open(img_path)
                ax = axes[class_idx, sample_idx] if num_classes > 1 else axes[sample_idx]
                ax.imshow(img)
                
                if sample_idx == 0:
                    ax.set_ylabel(label, fontsize=12, fontweight='bold')
                
                ax.axis('off')
            except Exception as e:
                ax = axes[class_idx, sample_idx] if num_classes > 1 else axes[sample_idx]
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                ax.axis('off')
    
    plt.suptitle('Samples by Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Class samples visualization saved to {save_path}")
    plt.close()


def check_dataset_quality(metadata_path, images_dir):
    """
    Check for potential issues in the dataset
    
    Args:
        metadata_path: path to dataset_metadata.csv
        images_dir: path to images directory
    """
    df = pd.read_csv(metadata_path)
    images_dir = Path(images_dir)
    
    print("\n" + "="*60)
    print("DATASET QUALITY CHECK")
    print("="*60)
    
    issues = []
    
    # Check for missing files
    print("\nChecking for missing files...")
    missing_files = []
    for _, row in df.iterrows():
        img_path = images_dir / row['filename']
        if not img_path.exists():
            missing_files.append(row['filename'])
    
    if missing_files:
        print(f"  ⚠️  Found {len(missing_files)} missing files")
        issues.append(f"{len(missing_files)} missing files")
    else:
        print("  ✓ All files present")
    
    # Check image dimensions
    print("\nChecking image dimensions...")
    dimensions = []
    for _, row in df.sample(n=min(100, len(df))).iterrows():
        img_path = images_dir / row['filename']
        try:
            img = Image.open(img_path)
            dimensions.append(img.size)
        except:
            pass
    
    if dimensions:
        unique_dims = set(dimensions)
        print(f"  Found {len(unique_dims)} unique dimensions")
        for dim in list(unique_dims)[:5]:
            print(f"    - {dim}")
        
        if len(unique_dims) > 10:
            issues.append("Many different image dimensions")
    
    # Check for duplicates
    print("\nChecking for duplicate filenames...")
    duplicates = df[df.duplicated(subset=['filename'], keep=False)]
    if len(duplicates) > 0:
        print(f"  ⚠️  Found {len(duplicates)} duplicate filenames")
        issues.append(f"{len(duplicates)} duplicate filenames")
    else:
        print("  ✓ No duplicate filenames")
    
    # Summary
    print("\n" + "="*60)
    if issues:
        print(f"⚠️  Found {len(issues)} potential issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Dataset quality check passed!")
    print("="*60)


def main():
    """Main analysis function"""
    
    # Paths
    metadata_path = 'unified_dataset/dataset_metadata.csv'
    images_dir = 'unified_dataset/images'
    analysis_dir = 'analysis'
    
    # Create analysis directory
    Path(analysis_dir).mkdir(exist_ok=True)
    
    # Run analyses
    print("\n" + "="*60)
    print("STARTING DATASET ANALYSIS")
    print("="*60)
    
    # 1. Analyze dataset statistics
    df = analyze_dataset(metadata_path, save_dir=analysis_dir)
    
    # 2. Visualize random samples
    print("\nGenerating sample visualizations...")
    visualize_samples(metadata_path, images_dir, num_samples=20, 
                     save_path=f'{analysis_dir}/random_samples.png')
    
    # 3. Visualize class samples
    visualize_class_samples(metadata_path, images_dir, num_classes=10, 
                           samples_per_class=5, 
                           save_path=f'{analysis_dir}/class_samples.png')
    
    # 4. Check dataset quality
    check_dataset_quality(metadata_path, images_dir)
    
    print("\n" + "="*60)
    print(f"✓ Analysis complete! Results saved to '{analysis_dir}/' directory")
    print("="*60)


if __name__ == "__main__":
    main()
