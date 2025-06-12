class ConversionError(Exception):
    """Base exception for conversion-related errors."""
    pass


class UnsupportedModelTypeError(ConversionError):
    """Raised when the model type is not supported for conversion."""
    pass


class ModelLoadError(ConversionError):
    """Raised when a model cannot be loaded."""
    pass


class ConversionTimeoutError(ConversionError):
    """Raised when conversion takes too long."""
    pass


class InsufficientMemoryError(ConversionError):
    """Raised when there's not enough memory for conversion."""
    pass


class OptimizationError(ConversionError):
    """Raised when model optimization fails."""
    pass