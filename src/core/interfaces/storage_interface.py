from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
from core.entities.conversion_result import ConversionResult


class IStorageService(ABC):
    """Interface for storage services."""
    
    @abstractmethod
    def save_file(self, content: bytes, filename: str, directory: Optional[str] = None) -> Path:
        """Save file to storage."""
        pass
    
    @abstractmethod
    def load_file(self, file_path: str) -> bytes:
        """Load file from storage."""
        pass
    
    @abstractmethod
    def delete_file(self, file_path: str) -> bool:
        """Delete file from storage."""
        pass
    
    @abstractmethod
    def list_files(self, directory: str, extension: Optional[str] = None) -> List[str]:
        """List files in directory."""
        pass
    
    @abstractmethod
    def save_conversion_result(self, result: ConversionResult) -> bool:
        """Save conversion result metadata."""
        pass
    
    @abstractmethod
    def load_conversion_history(self) -> List[ConversionResult]:
        """Load conversion history."""
        pass
    
    @abstractmethod
    def cleanup_temp_files(self) -> bool:
        """Clean up temporary files."""
        pass