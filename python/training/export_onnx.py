#!/usr/bin/env python3
"""
METABALL Virtual Ad - ONNX Export Script
Phase 2 Step 6: Export trained PyTorch model to ONNX format for C++ inference

This script converts a trained CameraPoseNet model to ONNX format,
enabling efficient inference in C++ using OpenCV DNN or ONNX Runtime.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from models import create_model


def export_to_onnx(
    model,
    output_path,
    input_shape=(1, 3, 224, 224),
    opset_version=11,
    dynamic_axes=True,
    verbose=True
):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        opset_version: ONNX opset version (default: 11 for compatibility)
        dynamic_axes: Enable dynamic batch size (default: True)
        verbose: Print conversion details (default: True)
        
    Returns:
        output_path: Path to saved ONNX model
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Move to same device as model
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    
    print(f"Exporting model to ONNX format...")
    print(f"  Input shape: {input_shape}")
    print(f"  Opset version: {opset_version}")
    print(f"  Dynamic axes: {dynamic_axes}")
    print(f"  Output path: {output_path}")
    
    # Define dynamic axes for variable batch size
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        dynamic_axes_dict = None
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # Optimize constant folding
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_dict,
        verbose=verbose
    )
    
    print(f"Model exported successfully to: {output_path}")
    
    return output_path


def verify_onnx_model(onnx_path, verbose=True):
    """
    Verify ONNX model is valid.
    
    Args:
        onnx_path: Path to ONNX model
        verbose: Print verification details (default: True)
        
    Returns:
        is_valid: True if model is valid, False otherwise
    """
    print(f"\nVerifying ONNX model: {onnx_path}")
    
    try:
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Check model validity
        onnx.checker.check_model(onnx_model)
        
        if verbose:
            print("  ONNX model is valid!")
            print(f"  IR version: {onnx_model.ir_version}")
            print(f"  Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
            
            # Print input/output shapes
            print("\n  Model Inputs:")
            for input_tensor in onnx_model.graph.input:
                print(f"    Name: {input_tensor.name}")
                shape = [d.dim_value if d.dim_value > 0 else 'dynamic' 
                        for d in input_tensor.type.tensor_type.shape.dim]
                print(f"    Shape: {shape}")
            
            print("\n  Model Outputs:")
            for output_tensor in onnx_model.graph.output:
                print(f"    Name: {output_tensor.name}")
                shape = [d.dim_value if d.dim_value > 0 else 'dynamic' 
                        for d in output_tensor.type.tensor_type.shape.dim]
                print(f"    Shape: {shape}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: ONNX model verification failed: {e}")
        return False


def test_onnx_inference(onnx_path, input_shape=(1, 3, 224, 224), verbose=True):
    """
    Test ONNX model inference using ONNX Runtime.
    
    Args:
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape for testing
        verbose: Print test details (default: True)
        
    Returns:
        output: Model output tensor
    """
    print(f"\nTesting ONNX Runtime inference...")
    
    try:
        # Create ONNX Runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(onnx_path, session_options)
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        if verbose:
            print(f"  Input name: {input_name}")
            print(f"  Output name: {output_name}")
        
        # Create random input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        outputs = session.run([output_name], {input_name: dummy_input})
        
        if verbose:
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {outputs[0].shape}")
            print(f"  Output sample: {outputs[0][0]}")
        
        print("  ONNX Runtime inference successful!")
        
        return outputs[0]
        
    except Exception as e:
        print(f"  ERROR: ONNX Runtime inference failed: {e}")
        return None


def compare_pytorch_onnx(
    pytorch_model,
    onnx_path,
    num_tests=10,
    input_shape=(1, 3, 224, 224),
    tolerance=1e-5
):
    """
    Compare PyTorch and ONNX model outputs for accuracy.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        num_tests: Number of random inputs to test
        input_shape: Input tensor shape
        tolerance: Maximum allowed difference
        
    Returns:
        max_diff: Maximum difference found
        passed: True if all tests passed, False otherwise
    """
    print(f"\nComparing PyTorch vs ONNX outputs...")
    print(f"  Number of tests: {num_tests}")
    print(f"  Tolerance: {tolerance}")
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Set PyTorch model to eval mode
    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device
    
    max_diff = 0.0
    all_passed = True
    
    with torch.no_grad():
        for i in range(num_tests):
            # Create random input
            test_input = np.random.randn(*input_shape).astype(np.float32)
            
            # PyTorch inference
            torch_input = torch.from_numpy(test_input).to(device)
            torch_output = pytorch_model(torch_input).cpu().numpy()
            
            # ONNX inference
            onnx_output = session.run([output_name], {input_name: test_input})[0]
            
            # Calculate difference
            diff = np.abs(torch_output - onnx_output).max()
            max_diff = max(max_diff, diff)
            
            if diff > tolerance:
                print(f"  Test {i+1}/{num_tests}: FAILED (diff={diff:.2e})")
                all_passed = False
            else:
                print(f"  Test {i+1}/{num_tests}: PASSED (diff={diff:.2e})")
    
    print(f"\nComparison complete:")
    print(f"  Maximum difference: {max_diff:.2e}")
    print(f"  Status: {'PASSED' if all_passed else 'FAILED'}")
    
    return max_diff, all_passed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Export PyTorch model to ONNX format'
    )
    
    # Model parameters
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PyTorch model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--model_arch',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'],
        help='Model architecture (default: resnet18)'
    )
    
    # ONNX export parameters
    parser.add_argument(
        '--output',
        type=str,
        default='./models/camera_pose_net.onnx',
        help='Output path for ONNX model (default: ./models/camera_pose_net.onnx)'
    )
    parser.add_argument(
        '--opset_version',
        type=int,
        default=11,
        help='ONNX opset version (default: 11 for broad compatibility)'
    )
    parser.add_argument(
        '--input_height',
        type=int,
        default=224,
        help='Input image height (default: 224)'
    )
    parser.add_argument(
        '--input_width',
        type=int,
        default=224,
        help='Input image width (default: 224)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for export (default: 1)'
    )
    parser.add_argument(
        '--dynamic_batch',
        action='store_true',
        help='Enable dynamic batch size (recommended)'
    )
    
    # Testing parameters
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run ONNX Runtime inference test'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare PyTorch vs ONNX outputs'
    )
    parser.add_argument(
        '--num_tests',
        type=int,
        default=10,
        help='Number of comparison tests (default: 10)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-5,
        help='Tolerance for comparison (default: 1e-5)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device for PyTorch model (default: cpu, recommended for export)'
    )
    
    return parser.parse_args()


def main():
    """Main export function."""
    args = parse_args()
    
    print("=" * 80)
    print("METABALL Virtual Ad - ONNX Export")
    print("=" * 80)
    
    # Device setup
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Create model
    print(f"\nCreating model: {args.model_arch}")
    model = create_model(args.model_arch, pretrained=False)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("  Loaded state dict")
    
    model.to(device)
    model.eval()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    print("\n" + "=" * 80)
    print("Exporting to ONNX")
    print("=" * 80)
    
    input_shape = (args.batch_size, 3, args.input_height, args.input_width)
    
    export_to_onnx(
        model,
        str(output_path),
        input_shape=input_shape,
        opset_version=args.opset_version,
        dynamic_axes=args.dynamic_batch,
        verbose=True
    )
    
    # Verify ONNX model
    print("\n" + "=" * 80)
    print("Verifying ONNX Model")
    print("=" * 80)
    
    is_valid = verify_onnx_model(str(output_path), verbose=True)
    
    if not is_valid:
        print("\nERROR: ONNX model verification failed!")
        return
    
    # Test ONNX Runtime inference
    if args.test:
        print("\n" + "=" * 80)
        print("Testing ONNX Runtime")
        print("=" * 80)
        
        test_onnx_inference(str(output_path), input_shape=input_shape, verbose=True)
    
    # Compare PyTorch vs ONNX
    if args.compare:
        print("\n" + "=" * 80)
        print("Comparing PyTorch vs ONNX")
        print("=" * 80)
        
        max_diff, passed = compare_pytorch_onnx(
            model,
            str(output_path),
            num_tests=args.num_tests,
            input_shape=input_shape,
            tolerance=args.tolerance
        )
    
    # Final summary
    print("\n" + "=" * 80)
    print("Export Complete")
    print("=" * 80)
    print(f"\nONNX model saved to: {output_path}")
    print(f"  Model architecture: {args.model_arch}")
    print(f"  Input shape: {input_shape}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic batch: {'Yes' if args.dynamic_batch else 'No'}")
    
    print("\nUsage in C++:")
    print("  1. Load with OpenCV DNN: cv::dnn::readNetFromONNX()")
    print("  2. Load with ONNX Runtime: Ort::Session()")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
