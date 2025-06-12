# 🔄 ML Model Converter

A robust, extensible Python tool for converting machine learning models from various frameworks to **TensorFlow Lite (TFLite)** format. Built with SOLID principles, featuring both web and command-line interfaces.

## ✨ Features

- **🔄 Multi-Framework Support**: Convert from PyTorch, TensorFlow, Keras, and ONNX
- **🏗️ SOLID Architecture**: Clean, maintainable, and extensible codebase  
- **🌐 Web Interface**: Easy-to-use Streamlit web application
- **🖥️ CLI Interface**: Powerful command-line tools for automation
- **📊 Conversion History**: Track and analyze all your conversions
- **⚙️ Model Optimization**: Multiple optimization levels and quantization options
- **🧪 Model Testing**: Validate converted models automatically
- **📈 Analytics**: Detailed conversion statistics and performance metrics

## 🔧 Supported Model Formats

| Framework | Extensions | Notes |
|-----------|------------|-------|
| **PyTorch** | `.pth`, `.pt` | State dict or complete models |
| **TensorFlow** | `.pb` | Frozen graphs, SavedModel |
| **Keras** | `.h5`, `.keras` | Sequential and Functional models |
| **ONNX** | `.onnx` | Cross-platform models |

### Conversion Pipeline

```
PyTorch → ONNX → TensorFlow → TFLite
TensorFlow → TFLite
Keras → TFLite  
ONNX → TensorFlow → TFLite
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (Python 3.11+ recommended)
- Git (for cloning repository)

### Installation

**🚀 One-Command Setup:**

```bash
# Linux/macOS
chmod +x install.sh && ./install.sh

# Windows (Command Prompt)
install.bat

# Windows (PowerShell)  
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install.ps1
```

> 📖 **Need help?** See the detailed [Installation Guide](INSTALL.md) for step-by-step instructions and troubleshooting.

### Usage Options

#### 🌐 Web Interface (Recommended for beginners)

```bash
# Easy launcher
python launch_web.py

# Or direct command
streamlit run src/web/app.py
```

Open your browser to `http://localhost:8501` and start converting!

#### 🖥️ Command Line Interface

```bash
# Easy launcher
python launch_cli.py --help

# Or direct commands
python launch_cli.py convert model.pth
python launch_cli.py validate model.pth
python launch_cli.py list-formats
```

#### 🧪 Test the Installation

```bash
python test_integration.py
```

## 📖 Detailed Usage

### Web Interface

The web interface provides a user-friendly way to convert models:

1. **Upload Model**: Drag and drop or browse for your model file
2. **Configure**: Set input shape, optimization level, and quantization options
3. **Convert**: Click convert and monitor progress
4. **Download**: Get your converted TFLite model
5. **Analyze**: View conversion statistics and history

**Supported Pages:**
- **🏠 Home**: Overview and getting started
- **🔄 Converter**: Main conversion interface
- **📊 History**: Track and analyze conversions
- **⚙️ Settings**: Configure default behaviors

### Command Line Interface

The CLI provides powerful automation capabilities:

#### Convert a Model
```bash
# Basic conversion
python launch_cli.py convert model.pth

# Custom output and settings
python launch_cli.py convert model.pth -o output.tflite -s 1,3,256,256

# With quantization disabled
python launch_cli.py convert model.h5 --no-quantization

# Aggressive optimization
python launch_cli.py convert model.onnx --optimization aggressive
```

#### Validate a Model
```bash
# Basic validation
python launch_cli.py validate model.pth

# Deep integrity check
python launch_cli.py validate model.onnx --check-integrity

# Verbose output
python launch_cli.py validate model.h5 --verbose
```

#### List Supported Formats
```bash
# Show all formats
python launch_cli.py list-formats

# Detailed information
python launch_cli.py list-formats --verbose

# Specific format
python launch_cli.py list-formats -t pytorch
```

## ⚙️ Configuration Options

### Optimization Levels
- **None**: No optimization (largest size, fastest conversion)
- **Default**: Standard optimization (recommended)
- **Aggressive**: Maximum optimization (smallest size, slower conversion)

### Quantization Types
- **FLOAT32**: Full precision (largest size, best accuracy)
- **FLOAT16**: Half precision (smaller size, minimal accuracy loss)
- **INT8**: Integer quantization (smallest size, requires representative dataset)

### Advanced Options
- **Custom Operations**: Support for custom TensorFlow operations
- **Experimental Converter**: Use latest TensorFlow Lite features
- **Representative Dataset**: For accurate INT8 quantization

## 📁 Project Structure

```
ml-model-converter/
├── src/                      # Source code
│   ├── core/                 # Core abstractions and entities
│   ├── converters/          # Model-specific converters
│   ├── services/            # Application services
│   ├── infrastructure/      # Storage, logging, config
│   ├── web/                 # Streamlit web interface
│   └── cli/                 # Command-line interface
├── requirements/            # Dependencies
├── outputs/                 # Runtime-generated (ignored by git)
│   ├── converted/          # Converted TFLite models
│   ├── temp/               # Temporary conversion files
│   └── logs/               # Application logs
├── config/                  # Configuration files
├── tests/                   # Test files
├── docs/                    # Documentation
├── launch_web.py           # Web interface launcher
├── launch_cli.py           # CLI launcher
└── test_integration.py     # Integration test
```

**📝 Note**: The `outputs/` directory and all model files (`.pth`, `.h5`, `.onnx`, `.tflite`, etc.) are automatically created at runtime and ignored by git to keep the repository clean.

## 🏗️ Architecture Overview

This project follows **SOLID principles** for clean, maintainable code:

### Core Components

- **Interfaces**: Abstract contracts for all major components
- **Entities**: Domain objects (ModelInfo, ConversionResult, etc.)
- **Converters**: Model-specific conversion implementations
- **Services**: Business logic orchestration
- **Infrastructure**: External concerns (storage, logging)

### Design Patterns

- **Factory Pattern**: For creating appropriate converters
- **Strategy Pattern**: For different conversion approaches
- **Dependency Injection**: For loose coupling
- **Repository Pattern**: For data persistence

## 🧪 Testing

### Run Integration Test
```bash
python test_integration.py
```

### Manual Testing
```bash
# Test with a sample PyTorch model
python launch_cli.py validate best_model.pth
python launch_cli.py convert best_model.pth -s 1,3,224,224
```

## 🔍 Troubleshooting

### Common Issues

#### "Model validation failed"
- Ensure the model file exists and is not corrupted
- Check if the file extension matches the model type
- Verify the input shape is correct

#### "Converter cannot handle this model type"
- Check supported formats with `list-formats` command
- Ensure all required dependencies are installed

#### "Conversion failed" 
- Try different optimization levels
- Disable quantization for debugging
- Check the error logs in `outputs/logs/`

#### PyTorch-specific Issues
- For state_dict models, ensure you have the model class definition
- Complete saved models (with `torch.save(model, path)`) work best
- Check if the model uses custom operations

### Getting Help

1. **Check the logs**: Look in `outputs/logs/converter.log`
2. **Run with verbose mode**: Add `--verbose` to CLI commands
3. **Test individual components**: Use the validation command
4. **Check dependencies**: Ensure all packages are installed correctly

## 📊 Performance Tips

### For Better Conversion Speed
- Use SSD storage for temporary files
- Close other applications to free memory
- Use smaller models for testing
- Enable experimental converter features

### For Smaller Models
- Enable quantization (INT8 for smallest size)
- Use aggressive optimization
- Provide representative datasets for INT8 quantization
- Remove unused operations if possible

### For Better Accuracy
- Disable quantization for maximum precision
- Use FLOAT32 inference type
- Validate converted models thoroughly
- Compare outputs between original and converted models

## 🛠️ Development

### Adding New Converters

1. **Create converter class** in `src/converters/new_framework/`
2. **Inherit from BaseConverter** and implement required methods
3. **Register in ConverterFactory** 
4. **Add tests** and documentation

### Extending the Web Interface

1. **Add new pages** in `src/web/pages/`
2. **Create reusable components** in `src/web/components/`
3. **Update navigation** in `src/web/app.py`

### Contributing

1. Follow SOLID principles
2. Add comprehensive tests
3. Update documentation
4. Use type hints throughout
5. Follow the established code style

## 📋 TODO / Roadmap

- [ ] **TensorFlow/Keras Converter**: Complete implementation
- [ ] **ONNX Converter**: Add direct ONNX support
- [ ] **Cloud Storage**: Add cloud storage backends
- [ ] **Batch Processing**: Convert multiple models at once
- [ ] **Model Comparison**: Compare original vs converted models
- [ ] **Docker Support**: Containerized deployment
- [ ] **Model Optimization**: Advanced optimization techniques
- [ ] **Web API**: REST API for programmatic access

## 👨‍💻 Author

**SamuraiSatoshi** *in collaboration with Claude Code A.I. Tool*
- GitHub: [@samuraisatoshi](https://github.com/samuraisatoshi/)
- Email: samuraisatoshi@cryptoworld.8shield.net
- Repository: [ML-Model-Converter](https://github.com/samuraisatoshi/ML-Model-Converter)

*Built with dedication for the ML community* 🚀

### 🤝 Development Collaboration
This project was developed through an innovative collaboration between:
- **SamuraiSatoshi** - Vision, requirements, and project leadership
- **Claude (Anthropic AI)** - Code architecture, implementation, and best practices

## 📄 License

**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**

This project is licensed under CC BY-NC 4.0, which means:

✅ **You CAN:**
- Use for personal projects
- Use for educational purposes
- Use for research and academic work
- Modify and distribute with attribution
- Use for non-profit organizations

❌ **You CANNOT:**
- Use for commercial purposes without permission
- Sell or monetize this software
- Remove attribution to the original author

📋 **You MUST:**
- Give credit to the original author
- Link back to this repository
- Include the license file
- Indicate any changes made

For commercial licensing, please contact: samuraisatoshi@cryptoworld.8shield.net

See [LICENSE](LICENSE) file for complete terms.

## 🤝 Support

For questions, issues, or contributions:

1. **Documentation**: Check the `docs/` directory
2. **Examples**: Look at `example_usage.py` and test files
3. **Issues**: Create an issue for bugs or feature requests
4. **Discussions**: For general questions and discussions

## 💰 Support the Project

Help keep this project free and growing! We accept crypto donations:

- **Bitcoin, Solana, BNB, XRP, USDT, USDC** - [View all addresses](cryptoDonation.md)
- **In the app**: Use the "💰 Donations" page for easy QR codes
- **Other support**: ⭐ Star the repo, 🐛 report bugs, 📝 contribute code

Every donation helps us add new features, fix bugs, and keep the tool completely free for everyone!

## 🎉 Acknowledgments

- **TensorFlow Team**: For TensorFlow Lite
- **PyTorch Team**: For the PyTorch framework
- **ONNX Community**: For the ONNX standard
- **Streamlit Team**: For the amazing web framework

---

**Ready to convert your models? 🚀**

```bash
# Quick start with web interface
python launch_web.py

# Or try the CLI
python launch_cli.py list-formats
python launch_cli.py convert your_model.pth
```

Happy converting! 🎯
