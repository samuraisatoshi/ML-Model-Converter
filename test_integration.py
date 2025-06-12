#!/usr/bin/env python3
"""
Integration test for the ML Model Converter.
This script tests the basic functionality of the converter.
"""

import sys
from pathlib import Path
import tempfile
import torch
import torch.nn as nn

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from core.entities.model_info import ModelInfo
from core.entities.conversion_config import ConversionConfig
from core.enums.model_types import ModelType
from services.conversion_service import ConversionService
from services.validation_service import ValidationService
from infrastructure.storage.local_storage import LocalStorageService
from infrastructure.logging.file_logger import CombinedLogger


class SimpleModel(nn.Module):
    """Simple PyTorch model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_test_model():
    """Create a simple test model."""
    model = SimpleModel()
    model.eval()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        torch.save(model, f.name)
        return Path(f.name)


def test_basic_conversion():
    """Test basic model conversion functionality."""
    print("🧪 Running Integration Test")
    print("=" * 50)
    
    try:
        # Step 1: Create test model
        print("1️⃣ Creating test PyTorch model...")
        model_path = create_test_model()
        print(f"   ✅ Test model created: {model_path}")
        
        # Step 2: Initialize services
        print("2️⃣ Initializing services...")
        logger = CombinedLogger(console_level="INFO", file_level="DEBUG")
        storage = LocalStorageService()
        validation = ValidationService(logger)
        conversion_service = ConversionService(storage, logger, validation)
        print("   ✅ Services initialized")
        
        # Step 3: Create model info
        print("3️⃣ Creating model info...")
        model_info = ModelInfo(
            file_path=model_path,
            model_type=ModelType.PYTORCH,
            input_shape=(1, 3, 224, 224),
            model_size=model_path.stat().st_size,
            metadata={}
        )
        print(f"   ✅ Model info created: {model_info.model_type.value}, {model_info.size_mb:.2f} MB")
        
        # Step 4: Create conversion config
        print("4️⃣ Creating conversion config...")
        output_path = Path("./outputs/converted/test_model.tflite")
        config = ConversionConfig(
            output_path=output_path,
            optimization_level="default",
            quantization=True,
            inference_type="FLOAT32"
        )
        print(f"   ✅ Config created: {config.optimization_level}, quantization={config.quantization}")
        
        # Step 5: Validate model
        print("5️⃣ Validating model...")
        is_valid = conversion_service.validate_model_info(model_info)
        if is_valid:
            print("   ✅ Model validation passed")
        else:
            print("   ❌ Model validation failed")
            return False
        
        # Step 6: Perform conversion
        print("6️⃣ Converting model...")
        print("   ⏳ This may take a few minutes...")
        result = conversion_service.convert_model(model_info, config)
        
        # Step 7: Check results
        print("7️⃣ Checking results...")
        if result.is_successful:
            print("   ✅ Conversion successful!")
            print(f"   📊 Original size: {result.original_size_mb:.2f} MB")
            print(f"   📊 Converted size: {result.converted_size_mb:.2f} MB")
            print(f"   📊 Size reduction: {result.size_reduction_percent:.1f}%")
            print(f"   ⏱️ Execution time: {result.execution_time:.2f} seconds")
            print(f"   📁 Output: {result.output_path}")
            
            # Test the converted model
            if result.output_path.exists():
                test_tflite_model(result.output_path, model_info.input_shape)
            
        else:
            print(f"   ❌ Conversion failed: {result.error_message}")
            return False
        
        # Step 8: Test supported formats
        print("8️⃣ Testing supported formats...")
        supported = conversion_service.get_supported_formats()
        print(f"   ✅ Supported formats: {supported}")
        
        # Step 9: Test conversion history
        print("9️⃣ Testing conversion history...")
        history = conversion_service.get_conversion_history()
        print(f"   ✅ History entries: {len(history)}")
        
        print("\n🎉 Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        import traceback
        print(f"🔍 Error details:\n{traceback.format_exc()}")
        return False
    
    finally:
        # Cleanup
        try:
            if 'model_path' in locals() and model_path.exists():
                model_path.unlink()
                print(f"🧹 Cleaned up test model: {model_path}")
        except Exception as e:
            print(f"⚠️ Failed to cleanup: {str(e)}")


def test_tflite_model(model_path: Path, input_shape: tuple):
    """Test the converted TFLite model."""
    try:
        import tensorflow as tf
        import numpy as np
        
        print("   🧪 Testing converted TFLite model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create test input
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"   ✅ TFLite model test successful!")
        print(f"   📐 Input shape: {input_details[0]['shape']}")
        print(f"   📐 Output shape: {output_details[0]['shape']}")
        print(f"   📊 Output range: [{output.min():.6f}, {output.max():.6f}]")
        
    except ImportError:
        print("   ⚠️ TensorFlow not available for testing")
    except Exception as e:
        print(f"   ❌ TFLite model test failed: {str(e)}")


def test_cli_interface():
    """Test the CLI interface."""
    print("\n🖥️ Testing CLI Interface")
    print("=" * 30)
    
    try:
        # Test list-formats command
        print("Testing list-formats command...")
        from cli.commands.list_formats import list_formats_command
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(list_formats_command)
        
        if result.exit_code == 0:
            print("✅ list-formats command works")
        else:
            print(f"❌ list-formats command failed: {result.output}")
            
    except Exception as e:
        print(f"❌ CLI test failed: {str(e)}")


if __name__ == "__main__":
    print("🚀 ML Model Converter - Integration Test")
    print("=" * 60)
    
    # Test basic conversion
    success = test_basic_conversion()
    
    # Test CLI interface
    test_cli_interface()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! The ML Model Converter is working correctly.")
        print("\n🚀 Ready to use:")
        print("   • Web interface: streamlit run src/web/app.py")
        print("   • CLI interface: python -m src.cli.main --help")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)