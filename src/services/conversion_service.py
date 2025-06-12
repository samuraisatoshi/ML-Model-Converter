from typing import Optional, List
from pathlib import Path
import time

from core.interfaces.converter_interface import IModelConverter
from core.interfaces.storage_interface import IStorageService
from core.interfaces.logger_interface import ILogger
from core.interfaces.validator_interface import IValidationService
from core.entities.model_info import ModelInfo
from core.entities.conversion_config import ConversionConfig
from core.entities.conversion_result import ConversionResult
from core.enums.conversion_status import ConversionStatus
from core.exceptions.conversion_exceptions import ConversionError
from core.exceptions.validation_exceptions import ValidationError
from converters.factory.converter_factory import ConverterFactory


class ConversionService:
    """Service for orchestrating model conversion operations."""
    
    def __init__(
        self,
        storage_service: IStorageService,
        logger: ILogger,
        validation_service: Optional[IValidationService] = None,
        converter_factory: Optional[ConverterFactory] = None
    ):
        self._storage = storage_service
        self._logger = logger
        self._validation = validation_service
        self._factory = converter_factory or ConverterFactory(logger)
    
    def convert_model(
        self, 
        model_info: ModelInfo, 
        config: ConversionConfig
    ) -> ConversionResult:
        """Orchestrate the complete model conversion process."""
        self._logger.info(f"Starting conversion: {model_info.filename}")
        
        try:
            # Step 1: Validate input
            if self._validation:
                self._logger.debug("Validating input model...")
                if not self._validation.validate_model_file(str(model_info.file_path)):
                    raise ValidationError("Model file validation failed")
                
                if not self._validation.validate_model_format(model_info):
                    raise ValidationError("Model format validation failed")
                
                if not self._validation.validate_conversion_config(config):
                    raise ValidationError("Conversion configuration validation failed")
            
            # Step 2: Get appropriate converter
            self._logger.debug(f"Getting converter for {model_info.model_type.value}")
            converter = self._factory.get_converter(model_info.model_type)
            
            # Step 3: Validate converter can handle this model
            if not converter.can_convert(model_info):
                raise ConversionError(f"Converter cannot handle this model type: {model_info.model_type.value}")
            
            # Step 4: Perform conversion
            self._logger.info("Executing conversion...")
            result = converter.convert(model_info, config)
            
            # Step 5: Save conversion result
            if result.is_successful:
                self._logger.info("Saving conversion result...")
                self._storage.save_conversion_result(result)
                self._logger.info(f"Conversion completed successfully: {result.output_path}")
            else:
                self._logger.error(f"Conversion failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            self._logger.error(f"Conversion service error: {str(e)}")
            
            # Create failed result
            return ConversionResult(
                output_path=config.output_path,
                status=ConversionStatus.FAILED,
                execution_time=0.0,
                original_size=model_info.model_size,
                converted_size=0,
                error_message=str(e)
            )
    
    def get_supported_formats(self) -> dict[str, List[str]]:
        """Get all supported model formats."""
        supported = {}
        for model_type, converter_name in self._factory.get_available_converters().items():
            supported[model_type.value] = self._factory.get_supported_formats(model_type)
        return supported
    
    def validate_model_info(self, model_info: ModelInfo) -> bool:
        """Validate model info before conversion."""
        try:
            # Check if converter exists for this model type
            if not self._factory.is_supported(model_info.model_type):
                self._logger.error(f"Unsupported model type: {model_info.model_type.value}")
                return False
            
            # Get converter and validate
            converter = self._factory.get_converter(model_info.model_type)
            return converter.validate_input(model_info)
            
        except Exception as e:
            self._logger.error(f"Model validation failed: {str(e)}")
            return False
    
    def get_conversion_history(self) -> List[ConversionResult]:
        """Get conversion history from storage."""
        try:
            return self._storage.load_conversion_history()
        except Exception as e:
            self._logger.error(f"Failed to load conversion history: {str(e)}")
            return []
    
    def cleanup_temp_files(self) -> bool:
        """Clean up temporary files."""
        try:
            return self._storage.cleanup_temp_files()
        except Exception as e:
            self._logger.error(f"Failed to cleanup temp files: {str(e)}")
            return False