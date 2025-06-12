from pathlib import Path
from typing import Optional

from core.interfaces.validator_interface import IValidationService
from core.interfaces.logger_interface import ILogger
from core.entities.model_info import ModelInfo
from core.entities.conversion_config import ConversionConfig
from core.enums.model_types import ModelType


class ValidationService(IValidationService):
    """Service for validating models and configurations."""
    
    def __init__(self, logger: Optional[ILogger] = None):
        self._logger = logger
    
    def _log(self, level: str, message: str) -> None:
        """Internal logging method."""
        if self._logger:
            getattr(self._logger, level)(message)
    
    def validate_model_file(self, file_path: str) -> bool:
        """Validate if the model file exists and is accessible."""
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                self._log("error", f"Model file does not exist: {file_path}")
                return False
            
            # Check if it's a file (not a directory)
            if not path.is_file():
                self._log("error", f"Path is not a file: {file_path}")
                return False
            
            # Check if file is readable
            if not path.stat().st_size > 0:
                self._log("error", f"Model file is empty: {file_path}")
                return False
            
            self._log("debug", f"Model file validation passed: {file_path}")
            return True
            
        except Exception as e:
            self._log("error", f"Model file validation error: {str(e)}")
            return False
    
    def validate_model_format(self, model_info: ModelInfo) -> bool:
        """Validate if the model format is supported."""
        try:
            # Check model type
            if not isinstance(model_info.model_type, ModelType):
                self._log("error", f"Invalid model type: {model_info.model_type}")
                return False
            
            # Check file extension matches model type
            expected_extensions = self._get_expected_extensions(model_info.model_type)
            if model_info.extension not in expected_extensions:
                self._log("error", 
                    f"File extension {model_info.extension} doesn't match model type {model_info.model_type.value}. "
                    f"Expected: {expected_extensions}")
                return False
            
            # Validate input shape
            if not self._validate_input_shape(model_info.input_shape, model_info.model_type):
                return False
            
            self._log("debug", f"Model format validation passed: {model_info.model_type.value}")
            return True
            
        except Exception as e:
            self._log("error", f"Model format validation error: {str(e)}")
            return False
    
    def validate_conversion_config(self, config: ConversionConfig) -> bool:
        """Validate if the conversion configuration is valid."""
        try:
            # Check output path
            if not config.output_path:
                self._log("error", "Output path is required")
                return False
            
            # Check output directory is writable
            output_dir = config.output_path.parent
            if not output_dir.exists():
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self._log("error", f"Cannot create output directory: {str(e)}")
                    return False
            
            # Validate optimization level
            valid_levels = ["none", "default", "aggressive"]
            if config.optimization_level not in valid_levels:
                self._log("error", f"Invalid optimization level: {config.optimization_level}")
                return False
            
            # Validate inference type
            valid_types = ["FLOAT32", "FLOAT16", "INT8", "INT16"]
            if config.inference_type not in valid_types:
                self._log("error", f"Invalid inference type: {config.inference_type}")
                return False
            
            # Validate input shapes if provided
            if config.input_shapes:
                for name, shape in config.input_shapes.items():
                    if not isinstance(shape, list) or len(shape) == 0:
                        self._log("error", f"Invalid input shape for {name}: {shape}")
                        return False
            
            self._log("debug", "Conversion config validation passed")
            return True
            
        except Exception as e:
            self._log("error", f"Conversion config validation error: {str(e)}")
            return False
    
    def validate_model_integrity(self, model_info: ModelInfo) -> bool:
        """Validate if the model file is not corrupted."""
        try:
            # Basic file integrity checks
            if not self.validate_model_file(str(model_info.file_path)):
                return False
            
            # Model type specific integrity checks
            if model_info.model_type == ModelType.PYTORCH:
                return self._validate_pytorch_integrity(model_info)
            elif model_info.model_type == ModelType.ONNX:
                return self._validate_onnx_integrity(model_info)
            elif model_info.model_type in [ModelType.TENSORFLOW, ModelType.KERAS]:
                return self._validate_tensorflow_integrity(model_info)
            
            # Default validation passed
            return True
            
        except Exception as e:
            self._log("error", f"Model integrity validation error: {str(e)}")
            return False
    
    def _get_expected_extensions(self, model_type: ModelType) -> list[str]:
        """Get expected file extensions for a model type."""
        extension_map = {
            ModelType.PYTORCH: ['.pth', '.pt'],
            ModelType.TENSORFLOW: ['.pb', '.savedmodel'],
            ModelType.KERAS: ['.h5', '.keras'],
            ModelType.ONNX: ['.onnx'],
            ModelType.TFLITE: ['.tflite']
        }
        return extension_map.get(model_type, [])
    
    def _validate_input_shape(self, input_shape: tuple, model_type: ModelType) -> bool:
        """Validate input shape for the given model type."""
        if not input_shape or len(input_shape) < 2:
            self._log("error", f"Invalid input shape: {input_shape}")
            return False
        
        # Check for reasonable dimensions
        for dim in input_shape:
            if not isinstance(dim, int) or dim <= 0:
                self._log("error", f"Invalid dimension in input shape: {dim}")
                return False
        
        # Model type specific validations
        if model_type in [ModelType.PYTORCH, ModelType.TENSORFLOW, ModelType.KERAS]:
            # Typically expect at least batch and feature dimensions
            if len(input_shape) < 2:
                self._log("error", f"Input shape too small for {model_type.value}: {input_shape}")
                return False
        
        return True
    
    def _validate_pytorch_integrity(self, model_info: ModelInfo) -> bool:
        """Validate PyTorch model integrity."""
        try:
            import torch
            # Try to load the model metadata
            checkpoint = torch.load(model_info.file_path, map_location='cpu')
            # If we can load it without error, it's likely valid
            return True
        except Exception as e:
            self._log("error", f"PyTorch model integrity check failed: {str(e)}")
            return False
    
    def _validate_onnx_integrity(self, model_info: ModelInfo) -> bool:
        """Validate ONNX model integrity."""
        try:
            import onnx
            # Try to load and check the model
            model = onnx.load(str(model_info.file_path))
            onnx.checker.check_model(model)
            return True
        except Exception as e:
            self._log("error", f"ONNX model integrity check failed: {str(e)}")
            return False
    
    def _validate_tensorflow_integrity(self, model_info: ModelInfo) -> bool:
        """Validate TensorFlow model integrity."""
        try:
            if model_info.extension == '.h5':
                import tensorflow as tf
                # Try to load Keras model
                tf.keras.models.load_model(str(model_info.file_path), compile=False)
            elif model_info.extension == '.pb':
                # For .pb files, basic file check should suffice
                pass
            return True
        except Exception as e:
            self._log("error", f"TensorFlow model integrity check failed: {str(e)}")
            return False