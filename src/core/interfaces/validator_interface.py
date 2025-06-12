from abc import ABC, abstractmethod
from core.entities.model_info import ModelInfo
from core.entities.conversion_config import ConversionConfig


class IValidationService(ABC):
    """Interface for validation services."""
    
    @abstractmethod
    def validate_model_file(self, file_path: str) -> bool:
        """Validate if the model file exists and is accessible."""
        pass
    
    @abstractmethod
    def validate_model_format(self, model_info: ModelInfo) -> bool:
        """Validate if the model format is supported."""
        pass
    
    @abstractmethod
    def validate_conversion_config(self, config: ConversionConfig) -> bool:
        """Validate if the conversion configuration is valid."""
        pass
    
    @abstractmethod
    def validate_model_integrity(self, model_info: ModelInfo) -> bool:
        """Validate if the model file is not corrupted."""
        pass