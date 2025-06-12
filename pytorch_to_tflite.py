import torch
import torch.nn as nn
import onnx
import tensorflow as tf
import onnx2tf
import numpy as np
import argparse
import os
from pathlib import Path


class ModelConverter:
    def __init__(self, model_path, output_dir="./converted_models"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get model name without extension
        self.model_name = Path(model_path).stem
        
    def load_pytorch_model(self, model_class=None, input_shape=(1, 3, 224, 224)):
        """
        Load PyTorch model from .pth file
        
        Args:
            model_class: PyTorch model class (if None, assumes state_dict only)
            input_shape: Input tensor shape for the model
        """
        print(f"Loading PyTorch model from {self.model_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        if model_class is not None:
            # If model class is provided, instantiate and load state_dict
            model = model_class()
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            # Assume the checkpoint contains the entire model
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model = checkpoint['model']
            elif isinstance(checkpoint, nn.Module):
                model = checkpoint
            else:
                raise ValueError("Cannot determine model structure. Please provide model_class.")
        
        model.eval()
        self.pytorch_model = model
        self.input_shape = input_shape
        return model
    
    def pytorch_to_onnx(self):
        """
        Convert PyTorch model to ONNX format
        """
        print("Converting PyTorch model to ONNX...")
        
        # Create dummy input
        dummy_input = torch.randn(*self.input_shape)
        
        # ONNX export path
        onnx_path = self.output_dir / f"{self.model_name}.onnx"
        
        # Export to ONNX
        torch.onnx.export(
            self.pytorch_model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"ONNX model saved to: {onnx_path}")
        self.onnx_path = onnx_path
        return onnx_path
    
    def onnx_to_tensorflow(self):
        """
        Convert ONNX model to TensorFlow SavedModel format
        """
        print("Converting ONNX model to TensorFlow...")
        
        # SavedModel export path
        savedmodel_path = self.output_dir / f"{self.model_name}_savedmodel"
        
        # Convert using onnx2tf
        onnx2tf.convert(
            input_onnx_file_path=str(self.onnx_path),
            output_folder_path=str(savedmodel_path),
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True
        )
        
        print(f"TensorFlow SavedModel saved to: {savedmodel_path}")
        self.savedmodel_path = savedmodel_path
        return savedmodel_path
    
    def tensorflow_to_tflite(self, quantize=False):
        """
        Convert TensorFlow SavedModel to TFLite format
        
        Args:
            quantize: Whether to apply quantization for smaller model size
        """
        print("Converting TensorFlow model to TFLite...")
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_saved_model(str(self.savedmodel_path))
        
        if quantize:
            print("Applying quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # converter.target_spec.supported_types = [tf.float16]  # Optional: use float16
        
        # Convert to TFLite
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_suffix = "_quantized" if quantize else ""
        tflite_path = self.output_dir / f"{self.model_name}{tflite_suffix}.tflite"
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to: {tflite_path}")
        self.tflite_path = tflite_path
        return tflite_path
    
    def convert_full_pipeline(self, model_class=None, input_shape=(1, 3, 224, 224), quantize=False):
        """
        Run the complete conversion pipeline: PyTorch → ONNX → TensorFlow → TFLite
        
        Args:
            model_class: PyTorch model class
            input_shape: Input tensor shape
            quantize: Whether to quantize the final TFLite model
        """
        try:
            # Step 1: Load PyTorch model
            self.load_pytorch_model(model_class, input_shape)
            
            # Step 2: Convert to ONNX
            self.pytorch_to_onnx()
            
            # Step 3: Convert to TensorFlow
            self.onnx_to_tensorflow()
            
            # Step 4: Convert to TFLite
            self.tensorflow_to_tflite(quantize)
            
            print(f"\n✅ Conversion completed successfully!")
            print(f"Final TFLite model: {self.tflite_path}")
            
            # Print model size
            model_size = os.path.getsize(self.tflite_path) / (1024 * 1024)
            print(f"Model size: {model_size:.2f} MB")
            
            return self.tflite_path
            
        except Exception as e:
            print(f"❌ Conversion failed: {str(e)}")
            raise e


def test_tflite_model(tflite_path, input_shape):
    """
    Test the converted TFLite model with random input
    """
    print(f"\nTesting TFLite model: {tflite_path}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Create test input
    test_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Inference successful! Output shape: {output.shape}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch models to TFLite')
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
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    # Create converter
    converter = ModelConverter(args.model_path, args.output_dir)
    
    # Convert model
    try:
        tflite_path = converter.convert_full_pipeline(
            input_shape=tuple(args.input_shape),
            quantize=args.quantize
        )
        
        # Test model if requested
        if args.test:
            test_tflite_model(tflite_path, tuple(args.input_shape))
            
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()

