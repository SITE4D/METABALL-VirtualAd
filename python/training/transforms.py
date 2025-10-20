"""
METABALL Virtual Ad - Data Augmentation Transforms
Augmentation strategies for camera pose estimation training.
"""

import random
from typing import Tuple, Optional

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class RandomBrightnessContrast:
    """Randomly adjust brightness and contrast."""
    
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), p=0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            # Adjust brightness
            brightness_factor = random.uniform(*self.brightness_range)
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)
            
            # Adjust contrast
            contrast_factor = random.uniform(*self.contrast_range)
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        
        return img


class RandomSaturationHue:
    """Randomly adjust saturation and hue."""
    
    def __init__(self, saturation_range=(0.8, 1.2), hue_range=(-0.1, 0.1), p=0.5):
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            # Adjust saturation
            saturation_factor = random.uniform(*self.saturation_range)
            img = ImageEnhance.Color(img).enhance(saturation_factor)
            
            # Adjust hue (convert to HSV, shift hue, convert back)
            if random.random() < 0.5:
                img_np = np.array(img).astype(np.float32) / 255.0
                # Simple hue shift approximation
                hue_shift = random.uniform(*self.hue_range)
                img_np = np.clip(img_np + hue_shift, 0, 1)
                img = Image.fromarray((img_np * 255).astype(np.uint8))
        
        return img


class RandomGaussianBlur:
    """Apply Gaussian blur with random kernel size."""
    
    def __init__(self, kernel_size_range=(3, 7), p=0.3):
        self.kernel_size_range = kernel_size_range
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            kernel_size = random.choice(range(
                self.kernel_size_range[0], 
                self.kernel_size_range[1] + 1, 
                2  # Only odd numbers
            ))
            img = img.filter(ImageFilter.GaussianBlur(radius=kernel_size // 2))
        
        return img


class RandomNoise:
    """Add random Gaussian noise."""
    
    def __init__(self, noise_std_range=(0, 0.05), p=0.3):
        self.noise_std_range = noise_std_range
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img_np = np.array(img).astype(np.float32) / 255.0
            noise_std = random.uniform(*self.noise_std_range)
            noise = np.random.normal(0, noise_std, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 1)
            img = Image.fromarray((img_np * 255).astype(np.uint8))
        
        return img


class RandomPerspective:
    """Apply random perspective transformation (mild for pose estimation)."""
    
    def __init__(self, distortion_scale=0.2, p=0.3):
        self.distortion_scale = distortion_scale
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            transform = T.RandomPerspective(
                distortion_scale=self.distortion_scale, 
                p=1.0
            )
            img = transform(img)
        
        return img


class RandomRotation:
    """Apply random rotation (mild for pose estimation)."""
    
    def __init__(self, degrees=5, p=0.3):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        return img


def get_training_transforms(
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    augmentation_level: str = 'medium'
) -> T.Compose:
    """
    Get training data transforms with augmentation.
    
    Args:
        target_size: Target image size (H, W)
        normalize: Whether to apply ImageNet normalization
        augmentation_level: 'light', 'medium', or 'heavy'
    
    Returns:
        torchvision.transforms.Compose object
    """
    transforms_list = []
    
    # Resize
    transforms_list.append(T.Resize(target_size))
    
    # Augmentation based on level
    if augmentation_level in ['medium', 'heavy']:
        # Color augmentations
        transforms_list.extend([
            RandomBrightnessContrast(p=0.5),
            RandomSaturationHue(p=0.5),
        ])
    
    if augmentation_level == 'heavy':
        # Geometric augmentations (mild for pose estimation)
        transforms_list.extend([
            RandomRotation(degrees=5, p=0.3),
            RandomPerspective(distortion_scale=0.15, p=0.3),
        ])
    
    if augmentation_level in ['medium', 'heavy']:
        # Noise and blur
        transforms_list.extend([
            RandomGaussianBlur(p=0.3),
            RandomNoise(p=0.2),
        ])
    
    # Convert to tensor
    transforms_list.append(T.ToTensor())
    
    # Normalization (ImageNet statistics)
    if normalize:
        transforms_list.append(
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    return T.Compose(transforms_list)


def get_validation_transforms(
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> T.Compose:
    """
    Get validation data transforms (no augmentation).
    
    Args:
        target_size: Target image size (H, W)
        normalize: Whether to apply ImageNet normalization
    
    Returns:
        torchvision.transforms.Compose object
    """
    transforms_list = [
        T.Resize(target_size),
        T.ToTensor()
    ]
    
    if normalize:
        transforms_list.append(
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    return T.Compose(transforms_list)


def get_inference_transforms(
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> T.Compose:
    """
    Get inference transforms (identical to validation).
    
    Args:
        target_size: Target image size (H, W)
        normalize: Whether to apply ImageNet normalization
    
    Returns:
        torchvision.transforms.Compose object
    """
    return get_validation_transforms(target_size, normalize)


class DenormalizeTransform:
    """Denormalize tensor for visualization."""
    
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: Normalized tensor of shape (3, H, W) or (B, 3, H, W)
        
        Returns:
            Denormalized tensor
        """
        if tensor.dim() == 4:
            # Batch dimension present
            mean = self.mean.unsqueeze(0).to(tensor.device)
            std = self.std.unsqueeze(0).to(tensor.device)
        else:
            mean = self.mean.to(tensor.device)
            std = self.std.to(tensor.device)
        
        return tensor * std + mean


# Preset configurations
AUGMENTATION_PRESETS = {
    'none': {
        'augmentation_level': 'light',
        'normalize': True
    },
    'light': {
        'augmentation_level': 'light',
        'normalize': True
    },
    'medium': {
        'augmentation_level': 'medium',
        'normalize': True
    },
    'heavy': {
        'augmentation_level': 'heavy',
        'normalize': True
    }
}


def get_transforms_from_preset(
    preset: str = 'medium',
    target_size: Tuple[int, int] = (224, 224),
    mode: str = 'train'
) -> T.Compose:
    """
    Get transforms from preset configuration.
    
    Args:
        preset: One of 'none', 'light', 'medium', 'heavy'
        target_size: Target image size (H, W)
        mode: 'train' or 'val'
    
    Returns:
        torchvision.transforms.Compose object
    """
    if preset not in AUGMENTATION_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(AUGMENTATION_PRESETS.keys())}")
    
    config = AUGMENTATION_PRESETS[preset]
    
    if mode == 'train':
        return get_training_transforms(
            target_size=target_size,
            normalize=config['normalize'],
            augmentation_level=config['augmentation_level']
        )
    else:
        return get_validation_transforms(
            target_size=target_size,
            normalize=config['normalize']
        )
