import json
import shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from core.interfaces.storage_interface import IStorageService
from core.entities.conversion_result import ConversionResult
from core.enums.conversion_status import ConversionStatus


class LocalStorageService(IStorageService):
    """Local filesystem storage implementation."""
    
    def __init__(
        self, 
        base_dir: str = "./outputs",
        converted_dir: str = "converted",
        temp_dir: str = "temp",
        logs_dir: str = "logs"
    ):
        self.base_path = Path(base_dir)
        self.converted_path = self.base_path / converted_dir
        self.temp_path = self.base_path / temp_dir
        self.logs_path = self.base_path / logs_dir
        self.history_file = self.base_path / "conversion_history.json"
        
        # Create directories
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for path in [self.base_path, self.converted_path, self.temp_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def save_file(self, content: bytes, filename: str, directory: Optional[str] = None) -> Path:
        """Save file to local storage."""
        if directory:
            target_dir = self.base_path / directory
        else:
            target_dir = self.converted_path
        
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return file_path
    
    def load_file(self, file_path: str) -> bytes:
        """Load file from local storage."""
        path = Path(file_path)
        with open(path, 'rb') as f:
            return f.read()
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file from local storage."""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception:
            return False
    
    def list_files(self, directory: str, extension: Optional[str] = None) -> List[str]:
        """List files in directory."""
        dir_path = self.base_path / directory
        if not dir_path.exists():
            return []
        
        files = []
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                if extension is None or file_path.suffix.lower() == extension.lower():
                    files.append(str(file_path))
        
        return sorted(files)
    
    def save_conversion_result(self, result: ConversionResult) -> bool:
        """Save conversion result metadata to history."""
        try:
            # Load existing history
            history = self.load_conversion_history()
            
            # Convert result to dict for JSON serialization
            result_dict = {
                "output_path": str(result.output_path),
                "status": result.status.value,
                "execution_time": result.execution_time,
                "original_size": result.original_size,
                "converted_size": result.converted_size,
                "compression_ratio": result.compression_ratio,
                "size_reduction_mb": result.size_reduction_mb,
                "size_reduction_percent": result.size_reduction_percent,
                "error_message": result.error_message,
                "timestamp": result.timestamp.isoformat(),
                "metadata": result.metadata
            }
            
            # Add to history
            history_data = [self._result_to_dict(r) for r in history] + [result_dict]
            
            # Save updated history
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
            
            return True
            
        except Exception:
            return False
    
    def load_conversion_history(self) -> List[ConversionResult]:
        """Load conversion history."""
        try:
            if not self.history_file.exists():
                return []
            
            with open(self.history_file, 'r') as f:
                history_data = json.load(f)
            
            results = []
            for item in history_data:
                try:
                    result = ConversionResult(
                        output_path=Path(item["output_path"]),
                        status=ConversionStatus(item["status"]),
                        execution_time=item["execution_time"],
                        original_size=item["original_size"],
                        converted_size=item["converted_size"],
                        metadata=item.get("metadata", {}),
                        error_message=item.get("error_message"),
                        timestamp=datetime.fromisoformat(item["timestamp"])
                    )
                    results.append(result)
                except Exception:
                    # Skip invalid entries
                    continue
            
            return results
            
        except Exception:
            return []
    
    def cleanup_temp_files(self) -> bool:
        """Clean up temporary files."""
        try:
            if self.temp_path.exists():
                shutil.rmtree(self.temp_path)
                self.temp_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
    
    def _result_to_dict(self, result: ConversionResult) -> dict:
        """Convert ConversionResult to dictionary."""
        return {
            "output_path": str(result.output_path),
            "status": result.status.value,
            "execution_time": result.execution_time,
            "original_size": result.original_size,
            "converted_size": result.converted_size,
            "compression_ratio": result.compression_ratio,
            "size_reduction_mb": result.size_reduction_mb,
            "size_reduction_percent": result.size_reduction_percent,
            "error_message": result.error_message,
            "timestamp": result.timestamp.isoformat(),
            "metadata": result.metadata
        }
    
    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        try:
            stats = {
                "total_converted_files": len(self.list_files("converted")),
                "total_temp_files": len(self.list_files("temp")),
                "total_log_files": len(self.list_files("logs")),
                "converted_dir_size": self._get_directory_size(self.converted_path),
                "temp_dir_size": self._get_directory_size(self.temp_path),
                "logs_dir_size": self._get_directory_size(self.logs_path)
            }
            return stats
        except Exception:
            return {}
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of a directory in bytes."""
        try:
            return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
        except Exception:
            return 0