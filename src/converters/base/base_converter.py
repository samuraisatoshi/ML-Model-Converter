from abc import ABC
from typing import List, Optional
import time
from pathlib import Path
from core.interfaces.converter_interface import IModelConverter
from core.interfaces.logger_interface import ILogger
from core.entities.model_info import ModelInfo
from core.entities.conversion_result import ConversionResult
from core.entities.conversion_config import ConversionConfig
from core.enums.conversion_status import ConversionStatus
from core.exceptions.conversion_exceptions import ConversionError


class BaseConverter(IModelConverter, ABC):
    """Base class for all model converters."""
    
    def __init__(self, logger: Optional[ILogger] = None):
        self._logger = logger
        
    def _log(self, level: str, message: str) -> None:
        """Internal logging method."""
        if self._logger:
            getattr(self._logger, level)(message)
    
    def validate_input(self, model_info: ModelInfo) -> bool:
        """Base validation for input models."""
        try:
            # Check if file exists
            if not model_info.file_path.exists():
                self._log("error", f"Model file not found: {model_info.file_path}")
                return False
            
            # Check file size
            if model_info.model_size <= 0:
                self._log("error", "Model file is empty")
                return False
            
            # Check if converter can handle this model type
            if not self.can_convert(model_info):
                self._log("error", f"Cannot convert model type: {model_info.model_type}")
                return False
            
            return True
            
        except Exception as e:
            self._log("error", f"Validation error: {str(e)}")
            return False
    
    def convert(self, model_info: ModelInfo, config: ConversionConfig) -> ConversionResult:
        """Base conversion method with timing and error handling."""
        start_time = time.time()
        
        try:
            self._log("info", f"Starting conversion: {model_info.filename}")
            
            # Validate input
            if not self.validate_input(model_info):
                raise ConversionError("Input validation failed")
            
            # Perform the actual conversion (implemented by subclasses)
            result = self._perform_conversion(model_info, config)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            self._log("info", f"Conversion completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._log("error", f"Conversion failed: {str(e)}")
            
            return ConversionResult(
                output_path=config.output_path,
                status=ConversionStatus.FAILED,
                execution_time=execution_time,
                original_size=model_info.model_size,
                converted_size=0,
                error_message=str(e)
            )
    
    def _perform_conversion(self, model_info: ModelInfo, config: ConversionConfig) -> ConversionResult:
        """Actual conversion logic to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _perform_conversion")
    
    def _create_success_result(
        self, 
        model_info: ModelInfo, 
        config: ConversionConfig, 
        output_path: Optional[Path] = None
    ) -> ConversionResult:
        """Helper method to create a successful conversion result."""
        output_path = output_path or config.output_path
        
        return ConversionResult(
            output_path=output_path,
            status=ConversionStatus.COMPLETED,
            execution_time=0.0,  # Will be set by base convert method
            original_size=model_info.model_size,
            converted_size=output_path.stat().st_size if output_path.exists() else 0
        )