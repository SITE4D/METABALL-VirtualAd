"""
Camera Pose Estimation Model
ResNet-18 based architecture for regressing camera pose (rvec, tvec) from images.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple


class CameraPoseNet(nn.Module):
    """
    Camera pose estimation network based on ResNet-18.
    
    Architecture:
        - Backbone: ResNet-18 (ImageNet pretrained)
        - Output: 6D vector (rvec[3] + tvec[3])
    
    Args:
        pretrained (bool): Use ImageNet pretrained weights
        dropout (float): Dropout rate before final layer
    """
    
    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super(CameraPoseNet, self).__init__()
        
        # Load ResNet-18 backbone
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final FC layer
        # ResNet-18 has 512 features before the final FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension from ResNet-18
        self.feature_dim = 512
        
        # Custom head for pose regression
        self.pose_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 6)  # 6D output: rvec (3) + tvec (3)
        )
        
        # Initialize the new layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for the custom layers."""
        for m in self.pose_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            pose: Camera pose [B, 6] (rvec[3] + tvec[3])
        """
        # Extract features with backbone
        features = self.backbone(x)  # [B, 512, 1, 1]
        
        # Regress pose
        pose = self.pose_head(features)  # [B, 6]
        
        return pose
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature vectors before pose regression.
        Useful for visualization and analysis.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            features: Feature vectors [B, 512]
        """
        features = self.backbone(x)  # [B, 512, 1, 1]
        features = torch.flatten(features, 1)  # [B, 512]
        return features


class EnsembleCameraPoseNet(nn.Module):
    """
    Ensemble of multiple CameraPoseNet models for improved accuracy.
    Predictions are averaged across all models.
    
    Args:
        num_models (int): Number of models in the ensemble
        pretrained (bool): Use pretrained weights for backbone
    """
    
    def __init__(self, num_models: int = 3, pretrained: bool = True):
        super(EnsembleCameraPoseNet, self).__init__()
        
        self.models = nn.ModuleList([
            CameraPoseNet(pretrained=pretrained) for _ in range(num_models)
        ])
        self.num_models = num_models
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all models and average predictions.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            pose: Averaged camera pose [B, 6]
        """
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Average predictions
        pose = torch.stack(predictions).mean(dim=0)
        return pose


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)) -> Dict:
    """
    Get model summary statistics.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    
    Returns:
        summary: Dictionary with model statistics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    summary = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'input_shape': (1,) + input_size,
        'output_shape': tuple(output.shape),
        'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
    }
    
    return summary


def print_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
    """
    Print formatted model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    
    summary = get_model_summary(model, input_size)
    
    print(f"Model Architecture: {model.__class__.__name__}")
    print(f"Input Shape:  {summary['input_shape']}")
    print(f"Output Shape: {summary['output_shape']}")
    print("-" * 80)
    print(f"Total Parameters:       {summary['total_params']:,}")
    print(f"Trainable Parameters:   {summary['trainable_params']:,}")
    print(f"Non-trainable Parameters: {summary['non_trainable_params']:,}")
    print(f"Estimated Size:         {summary['model_size_mb']:.2f} MB")
    print("=" * 80)


def create_model(model_type: str = 'single', pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Factory function to create camera pose estimation models.
    
    Args:
        model_type: Type of model ('single' or 'ensemble')
        pretrained: Use pretrained backbone weights
        **kwargs: Additional arguments for model initialization
    
    Returns:
        model: Initialized model
    """
    if model_type == 'single':
        return CameraPoseNet(pretrained=pretrained, **kwargs)
    elif model_type == 'ensemble':
        return EnsembleCameraPoseNet(pretrained=pretrained, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'single' or 'ensemble'.")


# Test and demonstration
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Camera Pose Estimation Model - Test Script")
    print("=" * 80 + "\n")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test 1: Single model
    print("Test 1: Single CameraPoseNet")
    print("-" * 80)
    model = CameraPoseNet(pretrained=True, dropout=0.5)
    model = model.to(device)
    model.eval()
    
    print_model_summary(model)
    
    # Test forward pass
    print("\nForward Pass Test:")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        features = model.extract_features(dummy_input)
    
    print(f"Input shape:    {dummy_input.shape}")
    print(f"Output shape:   {output.shape}")
    print(f"Features shape: {features.shape}")
    print(f"\nSample output (first batch):")
    print(f"  rvec: {output[0, :3].cpu().numpy()}")
    print(f"  tvec: {output[0, 3:].cpu().numpy()}")
    
    # Test 2: Ensemble model
    print("\n" + "=" * 80)
    print("Test 2: Ensemble CameraPoseNet")
    print("-" * 80)
    ensemble_model = EnsembleCameraPoseNet(num_models=3, pretrained=True)
    ensemble_model = ensemble_model.to(device)
    ensemble_model.eval()
    
    print_model_summary(ensemble_model)
    
    with torch.no_grad():
        ensemble_output = ensemble_model(dummy_input)
    
    print(f"\nEnsemble output shape: {ensemble_output.shape}")
    print(f"Sample ensemble output (first batch):")
    print(f"  rvec: {ensemble_output[0, :3].cpu().numpy()}")
    print(f"  tvec: {ensemble_output[0, 3:].cpu().numpy()}")
    
    # Test 3: Model creation factory
    print("\n" + "=" * 80)
    print("Test 3: Model Factory")
    print("-" * 80)
    
    factory_model = create_model('single', pretrained=False, dropout=0.3)
    print(f"Created model: {factory_model.__class__.__name__}")
    print(f"Pretrained: False, Dropout: 0.3")
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80 + "\n")
