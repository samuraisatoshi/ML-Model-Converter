from abc import ABC, abstractmethod
from typing import List
from core.entities.model_info import ModelInfo
from core.entities.conversion_result import ConversionResult
from core.entities.conversion_config import ConversionConfig


class IModelConverter(ABC):
    """Interface for model converters."""
    
    @abstractmethod
    def can_convert(self, model_info: ModelInfo) -> bool:
        """Check if this converter can handle the given model."""
        pass
    
    @abstractmethod
    def convert(self, model_info: ModelInfo, config: ConversionConfig) -> ConversionResult:
        """Convert the model to TensorFlow Lite format."""
        pass
    
    @abstractmethod
    def validate_input(self, model_info: ModelInfo) -> bool:
        """Validate if the input model is valid for conversion."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported model formats."""
        pass
    
    @abstractmethod
    def get_converter_name(self) -> str:
        """Get the name of this converter."""
        pass