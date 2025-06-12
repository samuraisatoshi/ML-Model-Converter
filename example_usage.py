#!/usr/bin/env python3
"""
Example usage of PyTorch to TFLite converter
"""

import torch
import torch.nn as nn
from pytorch_to_tflite import ModelConverter


# Example 1: Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_sample_model():
    """
    Create and save a sample PyTorch model for demonstration
    """
    print("Creating sample PyTorch model...")
    
    # Create model
    model = SimpleCNN(num_classes=10)
    
    # Create some dummy data and train for one step (just for demo)
    model.train()
    dummy_input = torch.randn(4, 3, 224, 224)
    dummy_target = torch.randint(0, 10, (4,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # One forward/backward pass
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    optimizer.step()
    
    # Save model
    model.eval()
    model_path = "sample_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Sample model saved to: {model_path}")
    
    return model_path, SimpleCNN


def example_conversion_with_model_class():
    """
    Example: Convert a model where you have the model class definition
    """
    print("\n" + "="*50)
    print("EXAMPLE 1: Conversion with Model Class")
    print("="*50)
    
    # Create sample model
    model_path, model_class = create_sample_model()
    
    # Create converter
    converter = ModelConverter(model_path, output_dir="./converted_models")
    
    # Convert with model class
    try:
        tflite_path = converter.convert_full_pipeline(
            model_class=model_class,
            input_shape=(1, 3, 224, 224),
            quantize=False
        )
        print(f"✅ Conversion successful! TFLite model: {tflite_path}")
        
        # Test the model
        from pytorch_to_tflite import test_tflite_model
        test_tflite_model(tflite_path, (1, 3, 224, 224))
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")


def example_conversion_full_model():
    """
    Example: Convert a model saved as complete model (not just state_dict)
    """
    print("\n" + "="*50)
    print("EXAMPLE 2: Conversion with Full Model")
    print("="*50)
    
    # Create and save complete model
    model = SimpleCNN(num_classes=5)
    model.eval()
    
    model_path = "complete_model.pth"
    torch.save(model, model_path)  # Save complete model
    print(f"Complete model saved to: {model_path}")
    
    # Create converter
    converter = ModelConverter(model_path, output_dir="./converted_models")
    
    # Convert without model class (model is complete)
    try:
        tflite_path = converter.convert_full_pipeline(
            model_class=None,  # No model class needed
            input_shape=(1, 3, 224, 224),
            quantize=True  # Apply quantization for smaller size
        )
        print(f"✅ Conversion successful! TFLite model: {tflite_path}")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")


def example_command_line_usage():
    """
    Show command line usage examples
    """
    print("\n" + "="*50)
    print("COMMAND LINE USAGE EXAMPLES")
    print("="*50)
    
    print("\n1. Basic conversion:")
    print("   python pytorch_to_tflite.py model.pth")
    
    print("\n2. With custom input shape:")
    print("   python pytorch_to_tflite.py model.pth --input_shape 1 1 28 28")
    
    print("\n3. With quantization:")
    print("   python pytorch_to_tflite.py model.pth --quantize")
    
    print("\n4. With testing:")
    print("   python pytorch_to_tflite.py model.pth --test")
    
    print("\n5. Custom output directory:")
    print("   python pytorch_to_tflite.py model.pth --output_dir ./my_models")
    
    print("\n6. Full options:")
    print("   python pytorch_to_tflite.py model.pth --input_shape 1 3 224 224 --quantize --test --output_dir ./output")


if __name__ == "__main__":
    print("PyTorch to TFLite Converter - Example Usage")
    print("===========================================")
    
    # Run examples
    example_conversion_with_model_class()
    example_conversion_full_model()
    example_command_line_usage()
    
    print("\n🎉 Examples completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run this example: python example_usage.py")
    print("3. Convert your own models: python pytorch_to_tflite.py your_model.pth")

