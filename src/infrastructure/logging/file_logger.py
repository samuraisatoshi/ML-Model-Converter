import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from core.interfaces.logger_interface import ILogger


class FileLogger(ILogger):
    """File-based logger implementation with rotation."""
    
    def __init__(
        self, 
        name: str = "MLConverter",
        log_file: str = "./outputs/logs/converter.log",
        level: str = "INFO",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        format_string: Optional[str] = None
    ):
        self.logger = logging.getLogger(name)
        
        # Set level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler if not exists
        if not self.logger.handlers:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(numeric_level)
            
            # Set formatter
            if format_string is None:
                format_string = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            
            formatter = logging.Formatter(format_string)
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
    
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
            # Create structured log entry
            extra_str = ", ".join([f"{k}={v}" for k, v in extra.items()])
            message = f"{message} | {extra_str}"
        
        self.logger.log(level, message)


class CombinedLogger(ILogger):
    """Logger that combines console and file logging."""
    
    def __init__(
        self,
        name: str = "MLConverter",
        log_file: str = "./outputs/logs/converter.log",
        console_level: str = "INFO",
        file_level: str = "DEBUG"
    ):
        self.console_logger = ConsoleLogger(f"{name}_console", console_level)
        self.file_logger = FileLogger(f"{name}_file", log_file, file_level)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message."""
        self.console_logger.debug(message, extra)
        self.file_logger.debug(message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message."""
        self.console_logger.info(message, extra)
        self.file_logger.info(message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message."""
        self.console_logger.warning(message, extra)
        self.file_logger.warning(message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message."""
        self.console_logger.error(message, extra)
        self.file_logger.error(message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message."""
        self.console_logger.critical(message, extra)
        self.file_logger.critical(message, extra)


# Import ConsoleLogger for backwards compatibility
from infrastructure.logging.console_logger import ConsoleLogger