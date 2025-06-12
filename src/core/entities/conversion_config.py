from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""
    output_path: Path
    optimization_level: str = "default"  # "none", "default", "aggressive"
    quantization: bool = True
    target_ops: Optional[List[str]] = None
    input_arrays: Optional[List[str]] = None
    output_arrays: Optional[List[str]] = None
    input_shapes: Optional[Dict[str, List[int]]] = None
    inference_type: str = "FLOAT32"  # "FLOAT32", "FLOAT16", "INT8"
    representative_dataset: Optional[Any] = None
    allow_custom_ops: bool = False
    experimental_new_converter: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Convert to Path object if string was passed
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate optimization level
        valid_levels = ["none", "default", "aggressive"]
        if self.optimization_level not in valid_levels:
            raise ValueError(f"Invalid optimization level: {self.optimization_level}")
        
        # Validate inference type
        valid_types = ["FLOAT32", "FLOAT16", "INT8", "INT16"]
        if self.inference_type not in valid_types:
            raise ValueError(f"Invalid inference type: {self.inference_type}")
    
    @property
    def output_filename(self) -> str:
        """Get the output filename."""
        return self.output_path.name
    
    @property
    def is_quantized(self) -> bool:
        """Check if quantization is enabled."""
        return self.quantization or self.inference_type in ["INT8", "INT16"]