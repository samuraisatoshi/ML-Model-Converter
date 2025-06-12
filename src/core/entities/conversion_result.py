from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from core.enums.conversion_status import ConversionStatus


@dataclass
class ConversionResult:
    """Result of a model conversion operation."""
    output_path: Path
    status: ConversionStatus
    execution_time: float
    original_size: int
    converted_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        # Convert to Path object if string was passed
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        
        # Calculate converted size if the file exists
        if self.output_path.exists() and self.converted_size <= 0:
            self.converted_size = self.output_path.stat().st_size
    
    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.original_size > 0:
            return self.converted_size / self.original_size
        return 0.0
    
    @property
    def size_reduction_mb(self) -> float:
        """Calculate size reduction in MB."""
        return (self.original_size - self.converted_size) / (1024 * 1024)
    
    @property
    def size_reduction_percent(self) -> float:
        """Calculate size reduction percentage."""
        if self.original_size > 0:
            return ((self.original_size - self.converted_size) / self.original_size) * 100
        return 0.0
    
    @property
    def is_successful(self) -> bool:
        """Check if conversion was successful."""
        return self.status == ConversionStatus.COMPLETED
    
    @property
    def original_size_mb(self) -> float:
        """Get original size in MB."""
        return self.original_size / (1024 * 1024)
    
    @property
    def converted_size_mb(self) -> float:
        """Get converted size in MB."""
        return self.converted_size / (1024 * 1024)