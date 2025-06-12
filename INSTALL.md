# 🚀 Installation Guide

Quick and easy installation for the ML Model Converter. Choose the method that works best for your system.

## ⚡ One-Command Installation

### Linux & macOS
```bash
chmod +x install.sh && ./install.sh
```

### Windows (Command Prompt)
```cmd
install.bat
```

### Windows (PowerShell)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install.ps1
```

## 📋 What the installer does

1. **Checks Python 3.10+** - Ensures compatible Python version
2. **Creates virtual environment** - Isolated Python environment
3. **Installs dependencies** - PyTorch, TensorFlow, Streamlit, ONNX
4. **Creates directories** - Sets up output folders
5. **Runs quick test** - Verifies installation

## 🎯 After Installation

### Start the Web Interface
```bash
python launch_web.py
# Opens http://localhost:8501
```

### Try the Command Line
```bash
python launch_cli.py --help
python launch_cli.py list-formats
```

### Test with your model
```bash
python launch_cli.py convert your_model.pth
```

## 🔧 Manual Installation (Advanced)

If the automated installer doesn't work for your setup:

1. **Install Python 3.10+** from [python.org](https://python.org)
2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements/base.txt
   pip install -r requirements/web.txt
   ```
4. **Create directories:**
   ```bash
   mkdir -p outputs/{converted,temp,logs}
   ```

## 🆘 Troubleshooting

### "Python not found"
- Install Python 3.10+ from [python.org](https://python.org)
- Make sure to check "Add Python to PATH"
- Restart your terminal/command prompt

### "Git not found" 
- Install Git from [git-scm.com](https://git-scm.com)
- Restart your terminal/command prompt

### Permission denied (Linux/Mac)
```bash
chmod +x install.sh
```

### PowerShell execution policy (Windows)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Dependencies fail to install
```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel
# Then try again
pip install -r requirements/base.txt
```

## ✅ Verify Installation

Run the integration test:
```bash
python test_integration.py
```

## 🌟 Ready to Convert!

Your ML Model Converter is now ready to use. Check out the [README.md](README.md) for detailed usage instructions.

**Quick start:**
- Web Interface: `python launch_web.py`
- Command Line: `python launch_cli.py convert model.pth`

Happy converting! 🎯

---

## 📄 License & Attribution

**ML Model Converter** by SamuraiSatoshi (in collaboration with Claude)  
Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)

- 🔗 **Repository**: https://github.com/samuraisatoshi/ML-Model-Converter
- 📧 **Contact**: samuraisatoshi@cryptoworld.8shield.net
- 💰 **Support**: See cryptoDonation.md for donation options

*Made with ❤️ for the ML community*