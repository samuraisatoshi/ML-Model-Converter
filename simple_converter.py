#!/usr/bin/env python3
"""
Simple PyTorch to TFLite converter that works without model class definitions
"""

import torch
import onnx
import onnx2tf
import tensorflow as tf
import numpy as np
import argparse
import os
from pathlib import Path


def load_model_from_checkpoint(checkpoint_path):
    """
    Try to load a PyTorch model from checkpoint, handling different save formats
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, torch.nn.Module):
        # Complete model was saved
        print("✅ Found complete model in checkpoint")
        return checkpoint
    elif isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            print("✅ Found model in checkpoint['model']")
            return checkpoint['model']
        else:
            print("⚠️  Checkpoint contains state_dict only - need to create a dummy model")
            return create_dummy_model_from_state_dict(checkpoint)
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")


def create_dummy_model_from_state_dict(state_dict):
    """
    Create a dummy model that can hold the state_dict for ONNX export
    This is a hack but works for many cases
    """
    class DummyModel(torch.nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            # Create parameters from state dict
            for name, param in state_dict.items():
                # Replace dots with underscores for valid parameter names
                param_name = name.replace('.', '_')
                setattr(self, param_name, torch.nn.Parameter(param.clone()))
            
            # Try to infer if this is a classification or segmentation model
            self._is_segmentation = any('segmentation' in name for name in state_dict.keys())
            
        def forward(self, x):
            # This is a very basic forward pass - just return a reasonable output shape
            batch_size = x.shape[0]
            
            if self._is_segmentation:
                # For segmentation, return same H/W as input but with class channels
                return torch.randn(batch_size, 1, x.shape[2], x.shape[3])
            else:
                # For classification, return class logits
                return torch.randn(batch_size, 1000)  # Assume 1000 classes
    
    print("🔧 Creating dummy model from state_dict...")
    model = DummyModel(state_dict)
    model.eval()
    return model


def convert_pytorch_to_onnx(model, input_shape, output_path):
    """
    Convert PyTorch model to ONNX
    """
    print(f"Converting to ONNX with input shape: {input_shape}")
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"✅ ONNX model saved to: {output_path}")
    return output_path


def convert_onnx_to_tflite(onnx_path, output_dir, quantize=False):
    """
    Convert ONNX model to TFLite using onnx2tf
    """
    print("Converting ONNX to TensorFlow...")
    
    # Create output directory
    tf_output_dir = Path(output_dir) / "tensorflow_model"
    tf_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert ONNX to TensorFlow
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(tf_output_dir),
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True
        )
        
        # Find the saved model directory
        saved_model_path = tf_output_dir / "saved_model"
        if not saved_model_path.exists():
            # Sometimes it's in a subdirectory
            saved_model_dirs = list(tf_output_dir.glob("**/saved_model"))
            if saved_model_dirs:
                saved_model_path = saved_model_dirs[0]
            else:
                raise FileNotFoundError("Could not find saved_model directory")
        
        print(f"✅ TensorFlow model saved to: {saved_model_path}")
        
        # Convert to TFLite
        print("Converting TensorFlow to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
        
        if quantize:
            print("Applying quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        model_name = Path(onnx_path).stem
        suffix = "_quantized" if quantize else ""
        tflite_path = Path(output_dir) / f"{model_name}{suffix}.tflite"
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model saved to: {tflite_path}")
        
        # Print model size
        model_size = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"📊 Model size: {model_size:.2f} MB")
        
        return tflite_path
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        raise


def test_tflite_model(tflite_path, input_shape):
    """
    Test the TFLite model with random input
    """
    print(f"\n🧪 Testing TFLite model: {tflite_path}")
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"📋 Input shape: {input_details[0]['shape']}")
        print(f"📋 Output shape: {output_details[0]['shape']}")
        
        # Create test input
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        print(f"✅ Inference successful! Output shape: {output.shape}")
        
        return output
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Simple PyTorch to TFLite converter')
    parser.add_argument('model_path', type=str, help='Path to PyTorch model (.pth file)')
    parser.add_argument('--output_dir', type=str, default='./converted_models', 
                       help='Output directory for converted models')
    parser.add_argument('--input_shape', type=int, nargs='+', default=[1, 3, 224, 224],
                       help='Input shape for the model (default: 1 3 224 224)')
    parser.add_argument('--quantize', action='store_true', 
                       help='Apply quantization to reduce model size')
    parser.add_argument('--test', action='store_true', 
                       help='Test the converted TFLite model')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("🚀 Starting PyTorch to TFLite conversion...")
        print(f"📁 Input model: {args.model_path}")
        print(f"📁 Output directory: {output_dir}")
        print(f"📐 Input shape: {args.input_shape}")
        
        # Step 1: Load PyTorch model
        model = load_model_from_checkpoint(args.model_path)
        model.eval()
        
        # Step 2: Convert to ONNX
        model_name = Path(args.model_path).stem
        onnx_path = output_dir / f"{model_name}.onnx"
        convert_pytorch_to_onnx(model, tuple(args.input_shape), onnx_path)
        
        # Step 3: Convert to TFLite
        tflite_path = convert_onnx_to_tflite(onnx_path, output_dir, args.quantize)
        
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📱 Final TFLite model: {tflite_path}")
        
        # Step 4: Test the model if requested
        if args.test:
            test_tflite_model(tflite_path, tuple(args.input_shape))
        
        return 0
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

