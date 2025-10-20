"""
METABALL Virtual Ad - Camera Pose Correction Dataset
Loads annotated backnet corners and camera parameters from JSON files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class BacknetCornersDataset(Dataset):
    """
    Dataset for loading backnet corner annotations and camera poses.
    
    Expected JSON format:
    {
        "frame_id": "frame_0001",
        "image_path": "path/to/frame_0001.jpg",
        "backnet_corners": [
            [x1, y1], [x2, y2], [x3, y3], [x4, y4]  # 4 corners in image space
        ],
        "camera_intrinsics": {
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": 960.0,
            "cy": 540.0,
            "dist_coeffs": [k1, k2, p1, p2, k3]
        },
        "camera_pose": {
            "rvec": [rx, ry, rz],  # Rodrigues rotation vector
            "tvec": [tx, ty, tz]   # translation vector
        }
    }
    """
    
    def __init__(
        self,
        annotation_dir: str,
        image_root: Optional[str] = None,
        transform=None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            annotation_dir: Directory containing JSON annotation files
            image_root: Root directory for images (if paths in JSON are relative)
            transform: Optional transforms to apply to images
            target_size: Target image size for network input (H, W)
        """
        self.annotation_dir = Path(annotation_dir)
        self.image_root = Path(image_root) if image_root else None
        self.transform = transform
        self.target_size = target_size
        
        # Load all annotation files
        self.annotations = self._load_annotations()
        
        if len(self.annotations) == 0:
            raise ValueError(f"No annotations found in {annotation_dir}")
        
        print(f"Loaded {len(self.annotations)} annotations from {annotation_dir}")
    
    def _load_annotations(self) -> List[Dict]:
        """Load all JSON annotation files from directory."""
        annotations = []
        
        json_files = sorted(self.annotation_dir.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Validate required fields
                required_fields = ["frame_id", "image_path", "backnet_corners", 
                                   "camera_intrinsics", "camera_pose"]
                if all(field in data for field in required_fields):
                    annotations.append(data)
                else:
                    print(f"Warning: Skipping {json_file.name} - missing required fields")
                    
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")
                continue
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict containing:
                - image: torch.Tensor of shape (3, H, W)
                - corners: torch.Tensor of shape (4, 2) - backnet corners
                - intrinsics: torch.Tensor of shape (4,) - [fx, fy, cx, cy]
                - rvec: torch.Tensor of shape (3,) - rotation vector
                - tvec: torch.Tensor of shape (3,) - translation vector
                - frame_id: str
        """
        annotation = self.annotations[idx]
        
        # Load image
        image_path = annotation["image_path"]
        if self.image_root and not Path(image_path).is_absolute():
            image_path = self.image_root / image_path
        
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (W, H)
        
        # Get backnet corners (4 points, each [x, y])
        corners = np.array(annotation["backnet_corners"], dtype=np.float32)
        
        # Scale corners if image is resized
        if self.target_size != (original_size[1], original_size[0]):
            scale_x = self.target_size[1] / original_size[0]  # target_W / orig_W
            scale_y = self.target_size[0] / original_size[1]  # target_H / orig_H
            corners[:, 0] *= scale_x
            corners[:, 1] *= scale_y
        
        # Resize image
        image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Get camera intrinsics
        intrinsics_dict = annotation["camera_intrinsics"]
        intrinsics = torch.tensor([
            intrinsics_dict["fx"],
            intrinsics_dict["fy"],
            intrinsics_dict["cx"],
            intrinsics_dict["cy"]
        ], dtype=torch.float32)
        
        # Get camera pose
        pose_dict = annotation["camera_pose"]
        rvec = torch.tensor(pose_dict["rvec"], dtype=torch.float32)
        tvec = torch.tensor(pose_dict["tvec"], dtype=torch.float32)
        
        return {
            "image": image,
            "corners": torch.from_numpy(corners),
            "intrinsics": intrinsics,
            "rvec": rvec,
            "tvec": tvec,
            "frame_id": annotation["frame_id"]
        }
    
    def get_statistics(self) -> Dict:
        """Calculate dataset statistics."""
        all_rvecs = []
        all_tvecs = []
        
        for ann in self.annotations:
            rvec = np.array(ann["camera_pose"]["rvec"])
            tvec = np.array(ann["camera_pose"]["tvec"])
            all_rvecs.append(rvec)
            all_tvecs.append(tvec)
        
        all_rvecs = np.array(all_rvecs)
        all_tvecs = np.array(all_tvecs)
        
        stats = {
            "num_samples": len(self.annotations),
            "rvec_mean": all_rvecs.mean(axis=0).tolist(),
            "rvec_std": all_rvecs.std(axis=0).tolist(),
            "tvec_mean": all_tvecs.mean(axis=0).tolist(),
            "tvec_std": all_tvecs.std(axis=0).tolist()
        }
        
        return stats


class CameraPoseDataset(Dataset):
    """
    Simple dataset for direct camera pose regression from images.
    Used for training the AI correction model.
    """
    
    def __init__(
        self,
        annotation_dir: str,
        image_root: Optional[str] = None,
        transform=None,
        normalize_pose: bool = True
    ):
        self.base_dataset = BacknetCornersDataset(
            annotation_dir, image_root, transform
        )
        self.normalize_pose = normalize_pose
        
        if normalize_pose:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Compute mean and std for pose normalization."""
        stats = self.base_dataset.get_statistics()
        self.rvec_mean = torch.tensor(stats["rvec_mean"])
        self.rvec_std = torch.tensor(stats["rvec_std"])
        self.tvec_mean = torch.tensor(stats["tvec_mean"])
        self.tvec_std = torch.tensor(stats["tvec_std"])
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: torch.Tensor of shape (3, H, W)
            pose: torch.Tensor of shape (6,) - concatenated [rvec, tvec]
        """
        sample = self.base_dataset[idx]
        image = sample["image"]
        
        # Concatenate rvec and tvec
        pose = torch.cat([sample["rvec"], sample["tvec"]], dim=0)
        
        # Normalize pose if enabled
        if self.normalize_pose:
            rvec_norm = (sample["rvec"] - self.rvec_mean) / (self.rvec_std + 1e-8)
            tvec_norm = (sample["tvec"] - self.tvec_mean) / (self.tvec_std + 1e-8)
            pose = torch.cat([rvec_norm, tvec_norm], dim=0)
        
        return image, pose


def create_data_loaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_root: Optional[str] = None,
    train_transform=None,
    val_transform=None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        train_dir: Directory with training annotations
        val_dir: Directory with validation annotations
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        image_root: Root directory for images
        train_transform: Transforms for training data
        val_transform: Transforms for validation data
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = CameraPoseDataset(
        train_dir, image_root, train_transform, normalize_pose=True
    )
    
    val_dataset = CameraPoseDataset(
        val_dir, image_root, val_transform, normalize_pose=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
