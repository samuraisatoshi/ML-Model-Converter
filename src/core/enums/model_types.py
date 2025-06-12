from enum import Enum


class ModelType(Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    KERAS = "keras"
    ONNX = "onnx"
    TFLITE = "tflite"
    
    @classmethod
    def from_extension(cls, file_path: str) -> "ModelType":
        """Determine model type from file extension."""
        extension = file_path.lower().split('.')[-1]
        
        extension_map = {
            'pth': cls.PYTORCH,
            'pt': cls.PYTORCH,
            'pb': cls.TENSORFLOW,
            'h5': cls.KERAS,
            'keras': cls.KERAS,
            'onnx': cls.ONNX,
            'tflite': cls.TFLITE,
        }
        
        if extension not in extension_map:
            raise ValueError(f"Unsupported model file extension: {extension}")
        
        return extension_map[extension]