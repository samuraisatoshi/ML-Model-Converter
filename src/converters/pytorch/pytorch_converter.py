from typing import List, Optional, Tuple, Any
import torch
import torch.nn as nn
import onnx
import tensorflow as tf
import onnx2tf
import numpy as np
from pathlib import Path
import tempfile
import shutil

from converters.base.base_converter import BaseConverter
from core.entities.model_info import ModelInfo
from core.entities.conversion_result import ConversionResult
from core.entities.conversion_config import ConversionConfig
from core.enums.model_types import ModelType
from core.exceptions.conversion_exceptions import (
    ModelLoadError, 
    ConversionError,
    UnsupportedModelTypeError
)


class PyTorchConverter(BaseConverter):
    """Converter for PyTorch models to TensorFlow Lite."""
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self._temp_dir = None
        
    def can_convert(self, model_info: ModelInfo) -> bool:
        """Check if this converter can handle PyTorch models."""
        return model_info.model_type == ModelType.PYTORCH
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported PyTorch formats."""
        return [".pth", ".pt"]
    
    def get_converter_name(self) -> str:
        """Get the name of this converter."""
        return "PyTorchConverter"
    
    def _perform_conversion(self, model_info: ModelInfo, config: ConversionConfig) -> ConversionResult:
        """Perform PyTorch to TFLite conversion."""
        try:
            self._temp_dir = tempfile.mkdtemp()
            temp_path = Path(self._temp_dir)
            
            # Step 1: Load PyTorch model
            self._log("info", "Loading PyTorch model...")
            pytorch_model = self._load_pytorch_model(model_info)
            
            # Step 2: Convert to ONNX
            self._log("info", "Converting PyTorch to ONNX...")
            onnx_path = self._pytorch_to_onnx(pytorch_model, model_info, temp_path)
            
            # Step 3: Convert ONNX to TensorFlow
            self._log("info", "Converting ONNX to TensorFlow...")
            savedmodel_path = self._onnx_to_tensorflow(onnx_path, temp_path)
            
            # Step 4: Convert TensorFlow to TFLite
            self._log("info", "Converting TensorFlow to TFLite...")
            self._tensorflow_to_tflite(savedmodel_path, config)
            
            return self._create_success_result(model_info, config)
            
        except Exception as e:
            self._log("error", f"Conversion failed: {str(e)}")
            raise ConversionError(f"PyTorch conversion failed: {str(e)}")
        finally:
            self._cleanup_temp_files()
    
    def _load_pytorch_model(self, model_info: ModelInfo) -> nn.Module:
        """Load PyTorch model from file with automatic structure detection."""
        try:
            # Load the checkpoint
            checkpoint = torch.load(model_info.file_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, nn.Module):
                # Case 1: Complete model saved with torch.save(model, path)
                model = checkpoint
                self._log("info", "Loaded complete PyTorch model")
            elif isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    # Case 2: Model saved in a dictionary under 'model' key
                    model = checkpoint['model']
                    self._log("info", "Loaded PyTorch model from 'model' key")
                elif 'state_dict' in checkpoint:
                    # Case 3: State dict with potential model architecture info
                    model = self._load_from_state_dict(checkpoint, model_info)
                    self._log("info", "Loaded PyTorch model from state_dict")
                else:
                    # Case 4: Assume checkpoint is the state_dict itself
                    model = self._load_from_state_dict({'state_dict': checkpoint}, model_info)
                    self._log("info", "Loaded PyTorch model from raw state_dict")
            else:
                raise ModelLoadError("Unsupported checkpoint format")
            
            model.eval()
            return model
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load PyTorch model: {str(e)}")
    
    def _load_from_state_dict(self, checkpoint: dict, model_info: ModelInfo) -> nn.Module:
        """Load model from state_dict with automatic architecture inference."""
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Strategy 1: Check if model_class is provided in metadata
        model_class = model_info.metadata.get('model_class')
        if model_class is not None:
            model = model_class()
            model.load_state_dict(state_dict)
            return model
        
        # Strategy 2: Try to extract architecture info from checkpoint metadata
        if 'arch' in checkpoint:
            model = self._create_model_from_arch_info(checkpoint['arch'], state_dict)
            if model is not None:
                return model
        
        # Strategy 3: Try to infer model structure from state_dict keys
        model = self._infer_model_from_state_dict(state_dict, model_info.input_shape)
        if model is not None:
            return model
        
        # Strategy 4: Try common model architectures
        model = self._try_common_architectures(state_dict, model_info.input_shape)
        if model is not None:
            return model
        
        raise ModelLoadError(
            "Cannot automatically determine model structure. Please provide model_class in ModelInfo.metadata. "
            "For best results, save models with torch.save(model, path) instead of just the state_dict."
        )
    
    def _create_model_from_arch_info(self, arch_info: any, state_dict: dict) -> Optional[nn.Module]:
        """Create model from architecture information in checkpoint."""
        try:
            if isinstance(arch_info, str):
                # Common architectures from torchvision
                import torchvision.models as models
                if hasattr(models, arch_info):
                    model_func = getattr(models, arch_info)
                    model = model_func()
                    model.load_state_dict(state_dict)
                    return model
        except Exception as e:
            self._log("debug", f"Failed to create model from arch info: {str(e)}")
        return None
    
    def _infer_model_from_state_dict(self, state_dict: dict, input_shape: tuple) -> Optional[nn.Module]:
        """Infer model structure from state_dict keys and create a compatible model."""
        try:
            # Analyze the state_dict structure
            layer_info = self._analyze_state_dict_structure(state_dict)
            
            # Try to create a simple model based on the analysis
            if layer_info['has_conv'] and layer_info['has_fc']:
                # Looks like a CNN
                model = self._create_simple_cnn(layer_info, input_shape)
                if model is not None:
                    try:
                        model.load_state_dict(state_dict, strict=False)
                        return model
                    except:
                        pass
            
        except Exception as e:
            self._log("debug", f"Failed to infer model from state_dict: {str(e)}")
        return None
    
    def _analyze_state_dict_structure(self, state_dict: dict) -> dict:
        """Analyze the structure of a state_dict to infer model architecture."""
        info = {
            'has_conv': False,
            'has_fc': False,
            'has_bn': False,
            'conv_layers': [],
            'fc_layers': [],
            'total_params': 0
        }
        
        for key, tensor in state_dict.items():
            if 'conv' in key.lower() and 'weight' in key:
                info['has_conv'] = True
                info['conv_layers'].append((key, tensor.shape))
            elif any(fc_name in key.lower() for fc_name in ['fc', 'linear', 'classifier']) and 'weight' in key:
                info['has_fc'] = True
                info['fc_layers'].append((key, tensor.shape))
            elif 'bn' in key.lower() or 'batch' in key.lower():
                info['has_bn'] = True
            
            info['total_params'] += tensor.numel()
        
        return info
    
    def _create_simple_cnn(self, layer_info: dict, input_shape: tuple) -> Optional[nn.Module]:
        """Create a simple CNN model based on layer analysis."""
        try:
            # This is a simplified approach - create a basic CNN structure
            class AutoInferredCNN(nn.Module):
                def __init__(self, input_channels, num_classes=1000):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(input_channels, 64, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((7, 7))
                    )
                    self.classifier = nn.Sequential(
                        nn.Dropout(),
                        nn.Linear(64 * 7 * 7, 4096),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(4096, num_classes)
                    )
                
                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x
            
            input_channels = input_shape[1] if len(input_shape) > 1 else 3
            
            # Try to infer number of classes from FC layers
            num_classes = 1000  # default
            if layer_info['fc_layers']:
                last_fc = layer_info['fc_layers'][-1]
                num_classes = last_fc[1][0]  # output dimension of last FC layer
            
            return AutoInferredCNN(input_channels, num_classes)
            
        except Exception as e:
            self._log("debug", f"Failed to create simple CNN: {str(e)}")
        return None
    
    def _try_common_architectures(self, state_dict: dict, input_shape: tuple) -> Optional[nn.Module]:
        """Try loading with common PyTorch model architectures."""
        try:
            import torchvision.models as models
            
            # List of common architectures to try
            common_models = [
                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                'vgg11', 'vgg13', 'vgg16', 'vgg19',
                'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                'densenet121', 'densenet161', 'densenet169', 'densenet201'
            ]
            
            for model_name in common_models:
                try:
                    if hasattr(models, model_name):
                        model_func = getattr(models, model_name)
                        model = model_func(pretrained=False)
                        
                        # Try to load the state_dict
                        model.load_state_dict(state_dict, strict=False)
                        
                        # Quick validation - check if most parameters were loaded
                        model_params = sum(p.numel() for p in model.parameters())
                        state_dict_params = sum(t.numel() for t in state_dict.values())
                        
                        if abs(model_params - state_dict_params) / max(model_params, state_dict_params) < 0.1:
                            self._log("info", f"Successfully matched with {model_name} architecture")
                            return model
                            
                except Exception:
                    continue
                    
        except Exception as e:
            self._log("debug", f"Failed to try common architectures: {str(e)}")
        
        return None
    
    def _pytorch_to_onnx(
        self, 
        model: nn.Module, 
        model_info: ModelInfo, 
        temp_path: Path
    ) -> Path:
        """Convert PyTorch model to ONNX format."""
        try:
            # Create dummy input
            dummy_input = torch.randn(*model_info.input_shape)
            
            # ONNX export path
            onnx_path = temp_path / f"{model_info.filename}.onnx"
            
            # Export to ONNX
            torch.onnx.export(
                model,
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
            
            return onnx_path
            
        except Exception as e:
            raise ConversionError(f"PyTorch to ONNX conversion failed: {str(e)}")
    
    def _onnx_to_tensorflow(self, onnx_path: Path, temp_path: Path) -> Path:
        """Convert ONNX model to TensorFlow SavedModel format."""
        try:
            # SavedModel export path
            savedmodel_path = temp_path / "savedmodel"
            
            # Convert using onnx2tf
            onnx2tf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(savedmodel_path),
                copy_onnx_input_output_names_to_tflite=True,
                non_verbose=True
            )
            
            return savedmodel_path
            
        except Exception as e:
            raise ConversionError(f"ONNX to TensorFlow conversion failed: {str(e)}")
    
    def _tensorflow_to_tflite(
        self, 
        savedmodel_path: Path, 
        config: ConversionConfig
    ) -> None:
        """Convert TensorFlow SavedModel to TFLite format."""
        try:
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_path))
            
            # Apply configuration
            self._apply_tflite_config(converter, config)
            
            # Convert to TFLite
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(config.output_path, 'wb') as f:
                f.write(tflite_model)
                
        except Exception as e:
            raise ConversionError(f"TensorFlow to TFLite conversion failed: {str(e)}")
    
    def _apply_tflite_config(
        self, 
        converter: tf.lite.TFLiteConverter, 
        config: ConversionConfig
    ) -> None:
        """Apply configuration settings to TFLite converter."""
        # Optimization settings
        if config.optimization_level == "aggressive":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif config.optimization_level == "default" and config.quantization:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Quantization settings
        if config.quantization:
            if config.inference_type == "INT8":
                converter.target_spec.supported_types = [tf.int8]
            elif config.inference_type == "FLOAT16":
                converter.target_spec.supported_types = [tf.float16]
        
        # Representative dataset for quantization
        if config.representative_dataset is not None:
            converter.representative_dataset = config.representative_dataset
        
        # Custom ops
        converter.allow_custom_ops = config.allow_custom_ops
        
        # Experimental settings
        converter.experimental_new_converter = config.experimental_new_converter
        
        # Target ops
        if config.target_ops:
            converter.target_spec.supported_ops = config.target_ops
    
    def _cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        if self._temp_dir and Path(self._temp_dir).exists():
            try:
                shutil.rmtree(self._temp_dir)
                self._log("debug", f"Cleaned up temporary directory: {self._temp_dir}")
            except Exception as e:
                self._log("warning", f"Failed to cleanup temp directory: {str(e)}")
    
    def validate_input(self, model_info: ModelInfo) -> bool:
        """Validate PyTorch model input."""
        if not super().validate_input(model_info):
            return False
        
        # Check if it's a PyTorch model
        if model_info.model_type != ModelType.PYTORCH:
            self._log("error", "Model is not a PyTorch model")
            return False
        
        # Check file extension
        if model_info.extension not in ['.pth', '.pt']:
            self._log("error", f"Unsupported PyTorch file extension: {model_info.extension}")
            return False
        
        # Check input shape
        if not model_info.input_shape or len(model_info.input_shape) < 2:
            self._log("error", "Invalid input shape for PyTorch model")
            return False
        
        return True