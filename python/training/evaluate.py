#!/usr/bin/env python3
"""
METABALL Virtual Ad - Model Evaluation Script
Phase 2 Step 5-3: Evaluation script for trained camera pose estimation model

This script evaluates a trained CameraPoseNet model on a test dataset,
calculating reprojection errors and generating visualizations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from dataset import CameraPoseDataset
from models import create_model
from transforms import get_inference_transforms


def load_checkpoint(checkpoint_path, model, device):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        device: Device to load model on
        
    Returns:
        Loaded model and checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}, val_loss: {val_loss}")
    else:
        # Assume checkpoint is just the state dict
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint (state dict only)")
    
    model.to(device)
    model.eval()
    
    return model, checkpoint


def run_inference(model, data_loader, device):
    """
    Run inference on dataset and collect predictions.
    
    Args:
        model: Trained model
        data_loader: DataLoader for test dataset
        device: Device to run inference on
        
    Returns:
        predictions: List of predicted poses [N, 6]
        ground_truths: List of ground truth poses [N, 6]
        image_paths: List of image paths
    """
    predictions = []
    ground_truths = []
    image_paths = []
    
    print("Running inference...")
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, poses) in enumerate(tqdm(data_loader, desc="Inference")):
            # Move to device
            images = images.to(device)
            
            # Forward pass
            pred_poses = model(images)
            
            # Collect results
            predictions.append(pred_poses.cpu().numpy())
            ground_truths.append(poses.numpy())
            
            # Note: image paths would need to be added to dataset __getitem__ return
            # For now, we'll skip this or handle separately
    
    # Concatenate all batches
    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)
    
    print(f"Inference complete: {len(predictions)} samples processed")
    
    return predictions, ground_truths


def calculate_reprojection_error(pred_poses, gt_poses, camera_matrix, object_points=None):
    """
    Calculate reprojection error between predicted and ground truth poses.
    
    Args:
        pred_poses: Predicted poses [N, 6] (rvec[3] + tvec[3])
        gt_poses: Ground truth poses [N, 6]
        camera_matrix: Camera intrinsic matrix [3, 3]
        object_points: 3D object points for reprojection [M, 3]
                      If None, uses unit cube corners
        
    Returns:
        errors: Reprojection errors per sample [N]
        statistics: Dictionary with mean, median, std, max errors
    """
    if object_points is None:
        # Default: unit cube corners for visualization
        object_points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ], dtype=np.float32)
    
    errors = []
    dist_coeffs = np.zeros(5)  # Assuming no distortion
    
    print("Calculating reprojection errors...")
    for i in tqdm(range(len(pred_poses)), desc="Reprojection"):
        # Extract rotation and translation vectors
        pred_rvec = pred_poses[i, :3].reshape(3, 1)
        pred_tvec = pred_poses[i, 3:].reshape(3, 1)
        gt_rvec = gt_poses[i, :3].reshape(3, 1)
        gt_tvec = gt_poses[i, 3:].reshape(3, 1)
        
        # Project 3D points using predicted pose
        pred_points_2d, _ = cv2.projectPoints(
            object_points,
            pred_rvec,
            pred_tvec,
            camera_matrix,
            dist_coeffs
        )
        
        # Project 3D points using ground truth pose
        gt_points_2d, _ = cv2.projectPoints(
            object_points,
            gt_rvec,
            gt_tvec,
            camera_matrix,
            dist_coeffs
        )
        
        # Calculate Euclidean distance (reprojection error)
        error = np.linalg.norm(pred_points_2d - gt_points_2d, axis=2).mean()
        errors.append(error)
    
    errors = np.array(errors)
    
    # Calculate statistics
    statistics = {
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
        'percentile_95': float(np.percentile(errors, 95)),
        'percentile_99': float(np.percentile(errors, 99))
    }
    
    print(f"\nReprojection Error Statistics (pixels):")
    print(f"  Mean:   {statistics['mean']:.2f}")
    print(f"  Median: {statistics['median']:.2f}")
    print(f"  Std:    {statistics['std']:.2f}")
    print(f"  Min:    {statistics['min']:.2f}")
    print(f"  Max:    {statistics['max']:.2f}")
    print(f"  95th percentile: {statistics['percentile_95']:.2f}")
    print(f"  99th percentile: {statistics['percentile_99']:.2f}")
    
    return errors, statistics


def calculate_pose_errors(pred_poses, gt_poses):
    """
    Calculate rotation and translation errors directly.
    
    Args:
        pred_poses: Predicted poses [N, 6]
        gt_poses: Ground truth poses [N, 6]
        
    Returns:
        rotation_errors: Rotation errors (angle in degrees) [N]
        translation_errors: Translation errors (Euclidean distance) [N]
        statistics: Dictionary with error statistics
    """
    rotation_errors = []
    translation_errors = []
    
    print("Calculating pose errors...")
    for i in tqdm(range(len(pred_poses)), desc="Pose errors"):
        # Rotation error (angle difference)
        pred_rvec = pred_poses[i, :3]
        gt_rvec = gt_poses[i, :3]
        
        # Convert to rotation matrices
        pred_R, _ = cv2.Rodrigues(pred_rvec)
        gt_R, _ = cv2.Rodrigues(gt_rvec)
        
        # Calculate rotation difference
        R_diff = np.dot(pred_R, gt_R.T)
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        rotation_errors.append(np.degrees(angle))
        
        # Translation error (Euclidean distance)
        pred_tvec = pred_poses[i, 3:]
        gt_tvec = gt_poses[i, 3:]
        t_error = np.linalg.norm(pred_tvec - gt_tvec)
        translation_errors.append(t_error)
    
    rotation_errors = np.array(rotation_errors)
    translation_errors = np.array(translation_errors)
    
    statistics = {
        'rotation': {
            'mean': float(np.mean(rotation_errors)),
            'median': float(np.median(rotation_errors)),
            'std': float(np.std(rotation_errors)),
            'max': float(np.max(rotation_errors))
        },
        'translation': {
            'mean': float(np.mean(translation_errors)),
            'median': float(np.median(translation_errors)),
            'std': float(np.std(translation_errors)),
            'max': float(np.max(translation_errors))
        }
    }
    
    print(f"\nRotation Error (degrees):")
    print(f"  Mean:   {statistics['rotation']['mean']:.2f}")
    print(f"  Median: {statistics['rotation']['median']:.2f}")
    print(f"  Std:    {statistics['rotation']['std']:.2f}")
    print(f"  Max:    {statistics['rotation']['max']:.2f}")
    
    print(f"\nTranslation Error:")
    print(f"  Mean:   {statistics['translation']['mean']:.4f}")
    print(f"  Median: {statistics['translation']['median']:.4f}")
    print(f"  Std:    {statistics['translation']['std']:.4f}")
    print(f"  Max:    {statistics['translation']['max']:.4f}")
    
    return rotation_errors, translation_errors, statistics


def visualize_predictions(pred_poses, gt_poses, errors, output_dir, num_samples=20):
    """
    Create visualizations comparing predictions to ground truth.
    
    Args:
        pred_poses: Predicted poses [N, 6]
        gt_poses: Ground truth poses [N, 6]
        errors: Reprojection errors [N]
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    
    # 1. Rotation vector comparison (rvec)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        axes[i].scatter(gt_poses[:, i], pred_poses[:, i], alpha=0.5, s=10)
        axes[i].plot([gt_poses[:, i].min(), gt_poses[:, i].max()],
                     [gt_poses[:, i].min(), gt_poses[:, i].max()],
                     'r--', label='Perfect prediction')
        axes[i].set_xlabel(f'Ground Truth rvec_{axis_name}')
        axes[i].set_ylabel(f'Predicted rvec_{axis_name}')
        axes[i].set_title(f'Rotation Vector {axis_name}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rotation_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'rotation_comparison.png'}")
    
    # 2. Translation vector comparison (tvec)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        axes[i].scatter(gt_poses[:, i+3], pred_poses[:, i+3], alpha=0.5, s=10)
        axes[i].plot([gt_poses[:, i+3].min(), gt_poses[:, i+3].max()],
                     [gt_poses[:, i+3].min(), gt_poses[:, i+3].max()],
                     'r--', label='Perfect prediction')
        axes[i].set_xlabel(f'Ground Truth tvec_{axis_name}')
        axes[i].set_ylabel(f'Predicted tvec_{axis_name}')
        axes[i].set_title(f'Translation Vector {axis_name}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'translation_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'translation_comparison.png'}")
    
    # 3. Error distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    ax.axvline(np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}')
    ax.set_xlabel('Reprojection Error (pixels)')
    ax.set_ylabel('Frequency')
    ax.set_title('Reprojection Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'error_distribution.png'}")
    
    # 4. Error vs sample index (to check for systematic errors)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(errors, alpha=0.6, linewidth=0.5)
    ax.axhline(np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Reprojection Error (pixels)')
    ax.set_title('Reprojection Error per Sample')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_per_sample.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'error_per_sample.png'}")
    
    print("Visualization complete!")


def save_detailed_results(pred_poses, gt_poses, errors, output_dir, num_samples=20):
    """
    Save detailed per-sample results to text file.
    
    Args:
        pred_poses: Predicted poses [N, 6]
        gt_poses: Ground truth poses [N, 6]
        errors: Reprojection errors [N]
        output_dir: Directory to save results
        num_samples: Number of samples to include in detailed output
    """
    output_dir = Path(output_dir)
    output_path = output_dir / 'detailed_results.txt'
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write(f"  Total samples: {len(pred_poses)}\n")
        f.write(f"  Mean error: {np.mean(errors):.4f} pixels\n")
        f.write(f"  Median error: {np.median(errors):.4f} pixels\n")
        f.write(f"  Std error: {np.std(errors):.4f} pixels\n")
        f.write(f"  Min error: {np.min(errors):.4f} pixels\n")
        f.write(f"  Max error: {np.max(errors):.4f} pixels\n\n")
        
        # Best and worst samples
        best_indices = np.argsort(errors)[:num_samples]
        worst_indices = np.argsort(errors)[-num_samples:][::-1]
        
        f.write(f"\nBest {num_samples} Predictions (lowest error):\n")
        f.write("-" * 80 + "\n")
        for idx in best_indices:
            f.write(f"Sample {idx}: Error = {errors[idx]:.4f} pixels\n")
            f.write(f"  Predicted: rvec={pred_poses[idx, :3]}, tvec={pred_poses[idx, 3:]}\n")
            f.write(f"  Ground Truth: rvec={gt_poses[idx, :3]}, tvec={gt_poses[idx, 3:]}\n\n")
        
        f.write(f"\nWorst {num_samples} Predictions (highest error):\n")
        f.write("-" * 80 + "\n")
        for idx in worst_indices:
            f.write(f"Sample {idx}: Error = {errors[idx]:.4f} pixels\n")
            f.write(f"  Predicted: rvec={pred_poses[idx, :3]}, tvec={pred_poses[idx, 3:]}\n")
            f.write(f"  Ground Truth: rvec={gt_poses[idx, :3]}, tvec={gt_poses[idx, 3:]}\n\n")
    
    print(f"Saved detailed results: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained camera pose estimation model'
    )
    
    # Model parameters
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--model_arch',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'],
        help='Model architecture (default: resnet18)'
    )
    
    # Data parameters
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to dataset directory containing images/ and annotations/'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    
    # Camera intrinsics (for reprojection error calculation)
    parser.add_argument(
        '--camera_matrix',
        type=str,
        default=None,
        help='Path to camera matrix JSON file (default: use identity)'
    )
    parser.add_argument(
        '--fx',
        type=float,
        default=800.0,
        help='Focal length x (default: 800.0)'
    )
    parser.add_argument(
        '--fy',
        type=float,
        default=800.0,
        help='Focal length y (default: 800.0)'
    )
    parser.add_argument(
        '--cx',
        type=float,
        default=320.0,
        help='Principal point x (default: 320.0)'
    )
    parser.add_argument(
        '--cy',
        type=float,
        default=240.0,
        help='Principal point y (default: 240.0)'
    )
    
    # Output parameters
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_results',
        help='Directory to save evaluation results (default: ./evaluation_results)'
    )
    parser.add_argument(
        '--save_visualizations',
        action='store_true',
        help='Save visualization images'
    )
    parser.add_argument(
        '--num_visualizations',
        type=int,
        default=20,
        help='Number of samples to visualize (default: 20)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation (default: cuda if available, else cpu)'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 80)
    print("METABALL Virtual Ad - Model Evaluation")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Device setup
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load or create camera matrix
    if args.camera_matrix and os.path.exists(args.camera_matrix):
        print(f"\nLoading camera matrix from: {args.camera_matrix}")
        with open(args.camera_matrix, 'r') as f:
            camera_data = json.load(f)
            camera_matrix = np.array(camera_data['camera_matrix'], dtype=np.float32)
    else:
        print(f"\nUsing default camera intrinsics:")
        print(f"  fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}")
        camera_matrix = np.array([
            [args.fx, 0, args.cx],
            [0, args.fy, args.cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    # Create dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    transforms = get_inference_transforms()
    dataset = CameraPoseDataset(
        data_dir=args.data_dir,
        transform=transforms
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create and load model
    print(f"\nCreating model: {args.model_arch}")
    model = create_model(args.model_arch, pretrained=False)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    model, checkpoint = load_checkpoint(args.checkpoint, model, device)
    
    # Run inference
    print("\n" + "=" * 80)
    print("Running Inference")
    print("=" * 80)
    pred_poses, gt_poses = run_inference(model, data_loader, device)
    
    # Calculate errors
    print("\n" + "=" * 80)
    print("Calculating Errors")
    print("=" * 80)
    
    # Reprojection error
    reproj_errors, reproj_stats = calculate_reprojection_error(
        pred_poses, gt_poses, camera_matrix
    )
    
    # Pose errors (rotation and translation)
    rot_errors, trans_errors, pose_stats = calculate_pose_errors(
        pred_poses, gt_poses
    )
    
    # Combine statistics
    all_statistics = {
        'reprojection': reproj_stats,
        'pose': pose_stats,
        'dataset_info': {
            'num_samples': len(dataset),
            'data_dir': args.data_dir
        },
        'model_info': {
            'architecture': args.model_arch,
            'checkpoint': args.checkpoint,
            'epoch': checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'unknown'
        }
    }
    
    # Save JSON report
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    json_path = output_dir / 'evaluation_report.json'
    with open(json_path, 'w') as f:
        json.dump(all_statistics, f, indent=2)
    print(f"Saved evaluation report: {json_path}")
    
    # Save detailed results
    save_detailed_results(
        pred_poses, gt_poses, reproj_errors, 
        output_dir, num_samples=args.num_visualizations
    )
    
    # Generate visualizations if requested
    if args.save_visualizations:
        visualize_predictions(
            pred_poses, gt_poses, reproj_errors, 
            output_dir, num_samples=args.num_visualizations
        )
    else:
        print("\nSkipping visualizations (use --save_visualizations to enable)")
    
    # Final summary
    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - evaluation_report.json: Overall statistics")
    print(f"  - detailed_results.txt: Per-sample analysis")
    if args.save_visualizations:
        print(f"  - rotation_comparison.png: Rotation predictions vs ground truth")
        print(f"  - translation_comparison.png: Translation predictions vs ground truth")
        print(f"  - error_distribution.png: Error histogram")
        print(f"  - error_per_sample.png: Error timeline")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
