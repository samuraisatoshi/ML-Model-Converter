from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from core.enums.model_types import ModelType


@dataclass
class ModelInfo:
    """Information about a model to be converted."""
    file_path: Path
    model_type: ModelType
    input_shape: Tuple[int, ...]
    model_size: int
    metadata: Dict[str, Any]
    framework_version: Optional[str] = None
    
    def __post_init__(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.file_path}")
        
        # Convert to Path object if string was passed
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
            
        # Get actual file size if not provided
        if self.model_size <= 0:
            self.model_size = self.file_path.stat().st_size
    
    @property
    def filename(self) -> str:
        """Get the model filename."""
        return self.file_path.name
    
    @property
    def extension(self) -> str:
        """Get the model file extension."""
        return self.file_path.suffix.lower()
    
    @property
    def size_mb(self) -> float:
        """Get model size in MB."""
        return self.model_size / (1024 * 1024)