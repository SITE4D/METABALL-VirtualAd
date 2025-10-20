"""
METABALL Virtual Ad - Data Loader Test Script
Test dataset loading and data augmentation transforms.
"""

import sys
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.dataset import BacknetCornersDataset, CameraPoseDataset, create_data_loaders
from training.transforms import (
    get_training_transforms,
    get_validation_transforms,
    DenormalizeTransform,
    get_transforms_from_preset
)


def visualize_sample(dataset, idx, denormalize=True):
    """Visualize a single sample from the dataset."""
    sample = dataset[idx]
    
    if isinstance(sample, tuple):
        # CameraPoseDataset returns (image, pose)
        image, pose = sample
        print(f"\nSample {idx}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Pose shape: {pose.shape}")
        print(f"  Pose values: {pose.numpy()}")
    else:
        # BacknetCornersDataset returns dict
        image = sample["image"]
        corners = sample["corners"]
        intrinsics = sample["intrinsics"]
        rvec = sample["rvec"]
        tvec = sample["tvec"]
        frame_id = sample["frame_id"]
        
        print(f"\nSample {idx} - {frame_id}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Corners shape: {corners.shape}")
        print(f"  Corners:\n{corners.numpy()}")
        print(f"  Intrinsics: {intrinsics.numpy()}")
        print(f"  rvec: {rvec.numpy()}")
        print(f"  tvec: {tvec.numpy()}")
    
    # Denormalize for visualization
    if denormalize and image.shape[0] == 3:
        denorm = DenormalizeTransform()
        image = denorm(image)
    
    # Convert to numpy for visualization
    img_np = image.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(img_np)
    
    # If we have corners, plot them
    if isinstance(sample, dict) and "corners" in sample:
        corners_np = corners.numpy()
        plt.scatter(corners_np[:, 0], corners_np[:, 1], c='red', s=100, marker='x')
        for i, (x, y) in enumerate(corners_np):
            plt.text(x, y, f'P{i}', color='yellow', fontsize=12, fontweight='bold')
    
    plt.title(f"Sample {idx}")
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()


def test_backnet_dataset(annotation_dir, image_root=None, num_samples=3):
    """Test BacknetCornersDataset."""
    print("=" * 60)
    print("Testing BacknetCornersDataset")
    print("=" * 60)
    
    # Test without transforms
    print("\n1. Testing without transforms...")
    dataset = BacknetCornersDataset(
        annotation_dir=annotation_dir,
        image_root=image_root,
        transform=None,
        target_size=(224, 224)
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Get statistics
    stats = dataset.get_statistics()
    print("\nDataset Statistics:")
    print(f"  Number of samples: {stats['num_samples']}")
    print(f"  rvec mean: {stats['rvec_mean']}")
    print(f"  rvec std: {stats['rvec_std']}")
    print(f"  tvec mean: {stats['tvec_mean']}")
    print(f"  tvec std: {stats['tvec_std']}")
    
    # Visualize samples
    num_samples = min(num_samples, len(dataset))
    for i in range(num_samples):
        fig = visualize_sample(dataset, i, denormalize=False)
        plt.savefig(f"test_output_sample_{i}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved visualization: test_output_sample_{i}.png")


def test_camera_pose_dataset(annotation_dir, image_root=None, num_samples=3):
    """Test CameraPoseDataset with transforms."""
    print("\n" + "=" * 60)
    print("Testing CameraPoseDataset with Transforms")
    print("=" * 60)
    
    # Test with training transforms
    print("\n2. Testing with training transforms (augmentation)...")
    train_transform = get_training_transforms(
        target_size=(224, 224),
        normalize=True,
        augmentation_level='medium'
    )
    
    dataset = CameraPoseDataset(
        annotation_dir=annotation_dir,
        image_root=image_root,
        transform=train_transform,
        normalize_pose=True
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Visualize augmented samples
    num_samples = min(num_samples, len(dataset))
    for i in range(num_samples):
        fig = visualize_sample(dataset, i, denormalize=True)
        plt.savefig(f"test_output_augmented_{i}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved augmented visualization: test_output_augmented_{i}.png")


def test_dataloader(annotation_dir, image_root=None, batch_size=4):
    """Test DataLoader with batching."""
    print("\n" + "=" * 60)
    print("Testing DataLoader")
    print("=" * 60)
    
    train_transform = get_transforms_from_preset('medium', target_size=(224, 224), mode='train')
    val_transform = get_transforms_from_preset('medium', target_size=(224, 224), mode='val')
    
    dataset = CameraPoseDataset(
        annotation_dir=annotation_dir,
        image_root=image_root,
        transform=train_transform,
        normalize_pose=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        pin_memory=False
    )
    
    print(f"\nDataLoader created with batch_size={batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Test iteration
    print("\nTesting batch iteration...")
    for batch_idx, (images, poses) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Poses shape: {poses.shape}")
        print(f"  Images min/max: {images.min():.3f} / {images.max():.3f}")
        print(f"  Poses min/max: {poses.min():.3f} / {poses.max():.3f}")
        
        if batch_idx >= 2:  # Test first 3 batches
            break
    
    print("\nDataLoader test completed successfully!")


def test_transform_presets():
    """Test different augmentation presets."""
    print("\n" + "=" * 60)
    print("Testing Transform Presets")
    print("=" * 60)
    
    presets = ['none', 'light', 'medium', 'heavy']
    
    for preset in presets:
        print(f"\nPreset: {preset}")
        train_transform = get_transforms_from_preset(preset, mode='train')
        val_transform = get_transforms_from_preset(preset, mode='val')
        print(f"  Train transforms: {len(train_transform.transforms)} steps")
        print(f"  Val transforms: {len(val_transform.transforms)} steps")


def main():
    parser = argparse.ArgumentParser(description="Test METABALL Virtual Ad data loader")
    parser.add_argument(
        "--annotation_dir",
        type=str,
        required=True,
        help="Directory containing JSON annotation files"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Root directory for images (if paths in JSON are relative)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for DataLoader test"
    )
    parser.add_argument(
        "--test_all",
        action="store_true",
        help="Run all tests"
    )
    
    args = parser.parse_args()
    
    # Check if annotation directory exists
    if not Path(args.annotation_dir).exists():
        print(f"Error: Annotation directory not found: {args.annotation_dir}")
        print("\nTo create sample annotations, use the annotation tool first.")
        return
    
    try:
        # Test transform presets
        test_transform_presets()
        
        # Test BacknetCornersDataset
        if args.test_all:
            test_backnet_dataset(args.annotation_dir, args.image_root, args.num_samples)
        
        # Test CameraPoseDataset
        test_camera_pose_dataset(args.annotation_dir, args.image_root, args.num_samples)
        
        # Test DataLoader
        test_dataloader(args.annotation_dir, args.image_root, args.batch_size)
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
