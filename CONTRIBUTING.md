# 🤝 Contributing to ML Model Converter

Thank you for your interest in contributing! This project follows SOLID principles and welcomes contributions from the community.

## 🚀 Quick Start for Contributors

1. **Fork and clone the repository**
2. **Install development dependencies:**
   ```bash
   # Use the quick installer
   ./install.sh  # Linux/macOS
   install.bat   # Windows
   
   # Install development tools
   pip install -r requirements/dev.txt
   ```
3. **Create a feature branch:**
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. **Make your changes and test:**
   ```bash
   python test_integration.py
   ```
5. **Submit a pull request**

## 📝 Contribution Types

### 🐛 Bug Fixes
- Fix issues in existing converters
- Improve error handling
- Fix documentation errors

### ✨ New Features
- **New Model Converters**: Add support for new frameworks
- **Web Interface**: New pages or components
- **CLI Commands**: Additional command-line functionality
- **Optimization**: Performance improvements

### 📚 Documentation
- Usage examples
- API documentation
- Tutorial content
- Translation

### 🧪 Testing
- Unit tests for new features
- Integration tests
- Edge case testing

## 🏗️ Architecture Guidelines

This project follows **SOLID principles**:

### Adding New Converters

1. **Create converter class** in `src/converters/new_framework/`
2. **Inherit from `BaseConverter`** and implement required methods:
   ```python
   from converters.base.base_converter import BaseConverter
   
   class MyFrameworkConverter(BaseConverter):
       def can_convert(self, model_info: ModelInfo) -> bool:
           # Implementation
       
       def convert(self, model_info: ModelInfo, config: ConversionConfig) -> ConversionResult:
           # Implementation
   ```
3. **Register in `ConverterFactory`** 
4. **Add tests** in `tests/unit/test_converters/`
5. **Update documentation**

### Code Standards

- **Type hints** are required for all public methods
- **Docstrings** for all classes and public methods
- **Error handling** with custom exceptions
- **Logging** for debugging and monitoring
- **Tests** for new functionality

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/unit/test_converters/

# Run integration test
python test_integration.py

# Test with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📋 Code Style

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check style
flake8 src/ tests/
mypy src/
```

## 🐛 Reporting Issues

When reporting bugs, please include:

1. **Python version**: `python --version`
2. **Operating system**: Windows/macOS/Linux
3. **Model file type**: PyTorch/TensorFlow/etc.
4. **Error message**: Full stack trace
5. **Steps to reproduce**: Minimal example

## 💡 Feature Requests

For new features, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** - why is this needed?
3. **Propose implementation** - how would it work?
4. **Consider SOLID principles** - how does it fit the architecture?

## 🔧 Development Setup

### Local Development

```bash
# Clone your fork
git clone https://github.com/yourusername/convertoToTFLite.git
cd convertoToTFLite

# Install in development mode
./install.sh
pip install -r requirements/dev.txt

# Set up pre-commit hooks (optional)
pre-commit install
```

### Testing Changes

```bash
# Test web interface
python launch_web.py

# Test CLI
python launch_cli.py list-formats
python launch_cli.py convert test_model.pth

# Run integration tests
python test_integration.py
```

## 📖 Documentation

When adding features:

1. **Update README.md** if it affects usage
2. **Add docstrings** to new classes/methods
3. **Update type hints** 
4. **Add examples** in docstrings
5. **Consider user guide updates**

## 🚀 Priority Areas

Help us improve these areas:

### High Priority
- **TensorFlow/Keras converter** improvements
- **ONNX converter** implementation
- **Model optimization** techniques
- **Error handling** improvements

### Medium Priority
- **Batch processing** multiple models
- **Cloud storage** backends
- **Model comparison** tools
- **Performance benchmarks**

### Low Priority
- **Docker support**
- **Web API** endpoints
- **Model visualizations**
- **Advanced quantization**

## 🤔 Questions?

- **Check the [README.md](README.md)** for basic usage
- **Read [CLAUDE.md](CLAUDE.md)** for architecture details
- **Browse existing issues** for similar questions
- **Open a discussion** for design questions

## 📄 License and Attribution

### License Terms
This project is licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

### By Contributing, You Agree That:
1. **Your contributions will be licensed** under the same CC BY-NC 4.0 license
2. **You have the right** to license your contributions under these terms
3. **You understand** that commercial use requires separate permission
4. **You will maintain attribution** to the original author in any derivatives

### Attribution Requirements
When contributing code, please:
- **Maintain existing copyright headers** in modified files
- **Add your name to contributors** if making significant changes
- **Reference the original repository** in any derivative works
- **Respect the non-commercial nature** of this project

### Commercial Use
- **Commercial use is prohibited** without explicit written permission
- **Contact the original author** for commercial licensing inquiries
- **Derivative works** must also be non-commercial unless licensed otherwise

### Code Headers
New files should include this header:
```python
#!/usr/bin/env python3
"""
File Description Here

Author: SamuraiSatoshi (Original), [Contributor Name] (Modifications)
License: CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
Repository: https://github.com/samuraisatoshi/ML-Model-Converter

Copyright (c) 2024 SamuraiSatoshi
This work is licensed under Creative Commons Attribution-NonCommercial 4.0 International.
For commercial licensing, contact: samuraisatoshi@cryptoworld.8shield.net
"""
```

---

**Thank you for making ML Model Converter better for everyone! 🎉**