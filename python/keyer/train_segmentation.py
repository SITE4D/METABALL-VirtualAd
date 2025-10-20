#!/usr/bin/env python3
"""
DeepLabV3+ Segmentation Training Script

野球映像のセグメンテーション（選手、審判、バックネット）を学習します。

使用方法:
    python train_segmentation.py --data_dir data/segmentation --epochs 50

依存関係:
    pip install torch torchvision albumentations segmentation-models-pytorch
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Segmentation Models PyTorch
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    print("WARNING: segmentation-models-pytorch not installed")
    print("  pip install segmentation-models-pytorch")
    SMP_AVAILABLE = False


class SegmentationDataset(Dataset):
    """セグメンテーションデータセット"""
    
    # クラス定義
    CLASS_NAMES = ['background', 'player', 'umpire', 'backnet']
    NUM_CLASSES = 4
    
    def __init__(self, data_dir: str, transform=None, split: str = 'train'):
        """
        初期化
        
        Args:
            data_dir: データディレクトリ
            transform: データ拡張変換
            split: 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # アノテーションファイルリスト取得
        self.annotations = []
        for json_file in sorted(self.data_dir.glob('*_annotation.json')):
            with open(json_file, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
                annotation['json_path'] = str(json_file)
                self.annotations.append(annotation)
        
        # Train/Val分割（80/20）
        split_idx = int(len(self.annotations) * 0.8)
        if split == 'train':
            self.annotations = self.annotations[:split_idx]
        else:
            self.annotations = self.annotations[split_idx:]
        
        print(f"Loaded {len(self.annotations)} annotations for {split} split")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        データ取得
        
        Returns:
            image: [3, H, W]
            mask: [H, W] (class indices)
        """
        annotation = self.annotations[idx]
        
        # 画像読み込み
        image_path = self.data_dir.parent / 'samples' / annotation['image']['file_name']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # マスク読み込み
        json_path = Path(annotation['json_path'])
        mask_path = json_path.parent / f"{json_path.stem.replace('_annotation', '')}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # データ拡張
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()


def get_training_transforms(image_size: int = 512):
    """学習用データ拡張"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_validation_transforms(image_size: int = 512):
    """検証用データ変換"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class DiceLoss(nn.Module):
    """Dice Loss"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W]
            target: [B, H, W]
        """
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Cross Entropy + Dice Loss"""
    
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[int, float]:
    """
    IoU計算
    
    Args:
        pred: [B, C, H, W]
        target: [B, H, W]
    
    Returns:
        各クラスのIoU
    """
    pred = torch.argmax(pred, dim=1)
    ious = {}
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union > 0:
            ious[cls] = (intersection / union).item()
        else:
            ious[cls] = 0.0
    
    return ious


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                   criterion: nn.Module, device: torch.device, epoch: int) -> Dict[str, float]:
    """1エポック学習"""
    model.train()
    
    total_loss = 0.0
    total_iou = {i: 0.0 for i in range(SegmentationDataset.NUM_CLASSES)}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        ious = calculate_iou(outputs, masks, SegmentationDataset.NUM_CLASSES)
        for cls, iou in ious.items():
            total_iou[cls] += iou
        
        # Progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Average metrics
    avg_loss = total_loss / len(dataloader)
    avg_iou = {cls: iou / len(dataloader) for cls, iou in total_iou.items()}
    mean_iou = np.mean(list(avg_iou.values()))
    
    return {
        'loss': avg_loss,
        'mean_iou': mean_iou,
        'class_ious': avg_iou
    }


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
            device: torch.device, epoch: int) -> Dict[str, float]:
    """検証"""
    model.eval()
    
    total_loss = 0.0
    total_iou = {i: 0.0 for i in range(SegmentationDataset.NUM_CLASSES)}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Metrics
            total_loss += loss.item()
            ious = calculate_iou(outputs, masks, SegmentationDataset.NUM_CLASSES)
            for cls, iou in ious.items():
                total_iou[cls] += iou
            
            # Progress bar
            pbar.set_postfix({'loss': loss.item()})
    
    # Average metrics
    avg_loss = total_loss / len(dataloader)
    avg_iou = {cls: iou / len(dataloader) for cls, iou in total_iou.items()}
    mean_iou = np.mean(list(avg_iou.values()))
    
    return {
        'loss': avg_loss,
        'mean_iou': mean_iou,
        'class_ious': avg_iou
    }


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int,
                   best_iou: float, save_path: str):
    """チェックポイント保存"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou
    }, save_path)
    print(f"Saved checkpoint: {save_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Train DeepLabV3+ for Baseball Segmentation')
    parser.add_argument('--data_dir', type=str, default='data/segmentation',
                       help='Data directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/segmentation',
                       help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    if not SMP_AVAILABLE:
        print("ERROR: segmentation-models-pytorch not installed")
        return
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # データセット
    train_dataset = SegmentationDataset(
        args.data_dir,
        transform=get_training_transforms(args.image_size),
        split='train'
    )
    val_dataset = SegmentationDataset(
        args.data_dir,
        transform=get_validation_transforms(args.image_size),
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # モデル（DeepLabV3+ with MobileNetV3 backbone）
    model = smp.DeepLabV3Plus(
        encoder_name='mobilenet_v2',
        encoder_weights='imagenet',
        classes=SegmentationDataset.NUM_CLASSES,
        activation=None  # Softmax is in loss
    )
    model = model.to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=f'runs/segmentation')
    
    # チェックポイントディレクトリ作成
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume
    start_epoch = 0
    best_iou = 0.0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['best_iou']
    
    # 学習ループ
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"Train Loss: {train_metrics['loss']:.4f} | Mean IoU: {train_metrics['mean_iou']:.4f}")
        
        # Validation
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        print(f"Val Loss: {val_metrics['loss']:.4f} | Mean IoU: {val_metrics['mean_iou']:.4f}")
        
        # クラス別IoU表示
        for cls in range(SegmentationDataset.NUM_CLASSES):
            class_name = SegmentationDataset.CLASS_NAMES[cls]
            train_iou = train_metrics['class_ious'][cls]
            val_iou = val_metrics['class_ious'][cls]
            print(f"  {class_name:10s}: Train IoU={train_iou:.4f} | Val IoU={val_iou:.4f}")
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('IoU/train', train_metrics['mean_iou'], epoch)
        writer.add_scalar('IoU/val', val_metrics['mean_iou'], epoch)
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_metrics['mean_iou'],
            checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        )
        
        # Save best model
        if val_metrics['mean_iou'] > best_iou:
            best_iou = val_metrics['mean_iou']
            save_checkpoint(
                model, optimizer, epoch, best_iou,
                checkpoint_dir / 'best_model.pth'
            )
            print(f"New best IoU: {best_iou:.4f}")
    
    writer.close()
    print("\nTraining completed!")
    print(f"Best validation IoU: {best_iou:.4f}")


if __name__ == '__main__':
    main()
