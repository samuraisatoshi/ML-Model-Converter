class ValidationError(Exception):
    """Base exception for validation-related errors."""
    pass


class InvalidModelFormatError(ValidationError):
    """Raised when the model format is invalid."""
    pass


class InvalidInputShapeError(ValidationError):
    """Raised when the input shape is invalid."""
    pass


class InvalidConfigurationError(ValidationError):
    """Raised when the conversion configuration is invalid."""
    pass


class FileNotFoundError(ValidationError):
    """Raised when the model file is not found."""
    pass


class CorruptedModelError(ValidationError):
    """Raised when the model file is corrupted."""
    pass