import logging
from typing import Any, Dict, Optional
from datetime import datetime

from core.interfaces.logger_interface import ILogger


class ConsoleLogger(ILogger):
    """Console-based logger implementation."""
    
    def __init__(
        self, 
        name: str = "MLConverter",
        level: str = "INFO",
        format_string: Optional[str] = None
    ):
        self.logger = logging.getLogger(name)
        
        # Set level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        
        # Create console handler if not exists
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(numeric_level)
            
            # Set formatter
            if format_string is None:
                format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            
            formatter = logging.Formatter(format_string)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message."""
        self._log(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, extra)
    
    def _log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Internal logging method."""
        if extra:
            # Append extra information to message
            extra_str = ", ".join([f"{k}={v}" for k, v in extra.items()])
            message = f"{message} | {extra_str}"
        
        self.logger.log(level, message)