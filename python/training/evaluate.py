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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Device setup
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # TODO: Step 5-3-2 - Load model and implement inference
    # TODO: Step 5-3-3 - Calculate reprojection errors
    # TODO: Step 5-3-4 - Generate visualizations
    # TODO: Step 5-3-5 - Main evaluation loop and report generation
    
    print("Evaluation script structure ready - implementation pending")


if __name__ == '__main__':
    main()
