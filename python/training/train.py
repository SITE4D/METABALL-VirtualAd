#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Camera Pose Estimation Model
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import project modules
from dataset import CameraPoseDataset
from transforms import get_train_transforms, get_val_transforms
from models import create_model


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10
) -> float:
    """
    Train the model for one epoch
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        log_interval: Log interval
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, poses) in enumerate(pbar):
        # Move data to device
        images = images.to(device)
        poses = poses.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_poses = model(images)
        loss = criterion(pred_poses, poses)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        
        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.6f}'
            })
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> float:
    """
    Validate the model
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        epoch: Current epoch number
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for images, poses in pbar:
            # Move data to device
            images = images.to(device)
            poses = poses.to(device)
            
            # Forward pass
            pred_poses = model(images)
            loss = criterion(pred_poses, poses)
            
            # Update statistics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss / (pbar.n + 1):.6f}'
            })
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_loss: float,
    checkpoint_path: Path,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        val_loss: Validation loss
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': val_loss if is_best else None
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.parent / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"[INFO] Saved best model to {best_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Camera Pose Estimation Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--train_annotations', type=str, default='annotations_train.json',
                        help='Training annotations filename')
    parser.add_argument('--val_annotations', type=str, default='annotations_val.json',
                        help='Validation annotations filename')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='resnet18',
                        choices=['resnet18', 'ensemble'],
                        help='Model architecture type')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained ImageNet weights')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='Validation batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    
    # Logging parameters
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log training status every N batches')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    
    args = parser.parse_args()
    return args


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Print configuration
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("=" * 80)
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"\nUsing GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("\nUsing CPU")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoint directory: {checkpoint_dir.absolute()}")
    
    # Initialize data loaders
    print("\n" + "=" * 80)
    print("Initializing Data Loaders")
    print("=" * 80)
    
    train_annotations_path = Path(args.data_dir) / args.train_annotations
    val_annotations_path = Path(args.data_dir) / args.val_annotations
    
    print(f"Train annotations: {train_annotations_path}")
    print(f"Val annotations: {val_annotations_path}")
    
    train_dataset = CameraPoseDataset(
        annotations=str(train_annotations_path),
        transform=get_train_transforms(),
        normalize_pose=True
    )
    
    val_dataset = CameraPoseDataset(
        annotations=str(val_annotations_path),
        transform=get_val_transforms(),
        normalize_pose=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
        drop_last=False
    )
    
    print(f"Train dataset: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val dataset: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Initialize model
    print("\n" + "=" * 80)
    print("Initializing Model")
    print("=" * 80)
    
    model = create_model(
        model_type=args.model_type,
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Print model summary
    from models import print_model_summary
    print_model_summary(model, device=device)
    
    # Initialize optimizer
    print("\n" + "=" * 80)
    print("Initializing Optimizer")
    print("=" * 80)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    print(f"Optimizer: Adam")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    
    # Initialize loss function
    criterion = nn.MSELoss()
    print(f"Loss function: MSELoss")
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print("\n" + "=" * 80)
        print("Loading Checkpoint")
        print("=" * 80)
        
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Resumed from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            log_interval=args.log_interval
        )
        
        # Validate
        val_loss = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch + 1
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  [NEW BEST] Validation loss improved to {best_val_loss:.6f}")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_freq == 0 or is_best:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1:04d}.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                checkpoint_path=checkpoint_path,
                is_best=is_best
            )
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir.absolute()}")


if __name__ == '__main__':
    main()
