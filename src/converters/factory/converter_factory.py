from typing import Dict, Optional
from core.interfaces.converter_interface import IModelConverter
from core.interfaces.logger_interface import ILogger
from core.enums.model_types import ModelType
from core.exceptions.conversion_exceptions import UnsupportedModelTypeError
from converters.pytorch.pytorch_converter import PyTorchConverter


class ConverterFactory:
    """Factory for creating model converters."""
    
    def __init__(self, logger: Optional[ILogger] = None):
        self._logger = logger
        self._converters: Dict[ModelType, IModelConverter] = {}
        self._initialize_converters()
    
    def _initialize_converters(self) -> None:
        """Initialize available converters."""
        # Register PyTorch converter
        self._converters[ModelType.PYTORCH] = PyTorchConverter(self._logger)
        
        # TODO: Add other converters when implemented
        # self._converters[ModelType.TENSORFLOW] = TensorFlowConverter(self._logger)
        # self._converters[ModelType.ONNX] = ONNXConverter(self._logger)
    
    def get_converter(self, model_type: ModelType) -> IModelConverter:
        """Get appropriate converter for the given model type."""
        if model_type not in self._converters:
            raise UnsupportedModelTypeError(f"No converter available for model type: {model_type.value}")
        
        return self._converters[model_type]
    
    def get_available_converters(self) -> Dict[ModelType, str]:
        """Get list of available converters."""
        return {
            model_type: converter.get_converter_name() 
            for model_type, converter in self._converters.items()
        }
    
    def is_supported(self, model_type: ModelType) -> bool:
        """Check if the model type is supported."""
        return model_type in self._converters
    
    def get_supported_formats(self, model_type: ModelType) -> list[str]:
        """Get supported file formats for a model type."""
        if model_type not in self._converters:
            return []
        return self._converters[model_type].get_supported_formats()
    
    def register_converter(self, model_type: ModelType, converter: IModelConverter) -> None:
        """Register a new converter for a model type."""
        self._converters[model_type] = converter
        if self._logger:
            self._logger.info(f"Registered converter for {model_type.value}: {converter.get_converter_name()}")
    
    def unregister_converter(self, model_type: ModelType) -> None:
        """Unregister a converter for a model type."""
        if model_type in self._converters:
            del self._converters[model_type]
            if self._logger:
                self._logger.info(f"Unregistered converter for {model_type.value}")