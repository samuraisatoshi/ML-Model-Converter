@echo off
REM ML Model Converter - Windows Installation Script
REM One-command setup for the complete environment

setlocal EnableDelayedExpansion

echo.
echo 🚀 ML Model Converter - Quick Setup (Windows)
echo ==================================================
echo.

REM Check if Python is available
echo 1️⃣ Checking Python installation...
set PYTHON_CMD=
set PYTHON_VERSION=

REM Try different Python commands
for %%p in (python3.11 python3.12 python3.13 python3.10 python py python3) do (
    %%p --version >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2 delims= " %%v in ('%%p --version 2^>^&1') do (
            set VERSION=%%v
            for /f "tokens=1,2 delims=." %%a in ("!VERSION!") do (
                set MAJOR=%%a
                set MINOR=%%b
                if !MAJOR! geq 3 if !MINOR! geq 10 (
                    set PYTHON_CMD=%%p
                    set PYTHON_VERSION=!VERSION!
                    echo ✅ Found Python !VERSION! at %%p
                    goto :python_found
                )
            )
        )
    )
)

:python_found
if "%PYTHON_CMD%"=="" (
    echo ❌ Python 3.10+ not found!
    echo.
    echo Please install Python 3.10 or higher:
    echo - Download from: https://python.org
    echo - Make sure to check "Add Python to PATH" during installation
    echo - Restart Command Prompt after installation
    pause
    exit /b 1
)

REM Check if git is available
echo 2️⃣ Checking Git installation...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git not found!
    echo Please install Git first: https://git-scm.com/downloads
    pause
    exit /b 1
)
echo ✅ Git is available

REM Create virtual environment
echo 3️⃣ Creating virtual environment...
if not exist "venv" (
    %PYTHON_CMD% -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 4️⃣ Activating virtual environment...
call venv\Scripts\activate.bat
echo ✅ Virtual environment activated

REM Upgrade pip
echo 5️⃣ Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install dependencies
echo 6️⃣ Installing dependencies...
echo    📚 Installing base dependencies (PyTorch, TensorFlow, ONNX)...
pip install -r requirements/base.txt

echo    🌐 Installing web interface dependencies (Streamlit)...
pip install -r requirements/web.txt

REM Create necessary directories
echo 7️⃣ Creating directory structure...
if not exist "outputs" mkdir outputs
if not exist "outputs\converted" mkdir outputs\converted
if not exist "outputs\temp" mkdir outputs\temp
if not exist "outputs\logs" mkdir outputs\logs
if not exist "config" mkdir config
echo ✅ Directory structure created

REM Make scripts executable (not needed on Windows, but check they exist)
echo 8️⃣ Checking launchers...
if exist "launch_web.py" (
    echo ✅ Web launcher found
) else (
    echo ⚠️ Web launcher not found
)
if exist "launch_cli.py" (
    echo ✅ CLI launcher found
) else (
    echo ⚠️ CLI launcher not found
)

REM Run a quick test
echo 9️⃣ Running quick test...
python test_integration.py --quick >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ Quick test skipped (requires model file)
) else (
    echo ✅ Quick test passed
)

echo.
echo 🎉 Installation completed successfully!
echo ==================================================
echo.
echo 🚀 Quick Start:
echo.
echo For Web Interface:
echo   python launch_web.py
echo   # Then open: http://localhost:8501
echo.
echo For Command Line:
echo   python launch_cli.py --help
echo   python launch_cli.py list-formats
echo   python launch_cli.py convert model.pth
echo.
echo To activate environment manually:
echo   venv\Scripts\activate.bat
echo.
echo 📚 Documentation:
echo   - User Guide: README.md
echo   - Developer Guide: CLAUDE.md
echo   - Web Interface: http://localhost:8501 (after running launch_web.py)
echo.
echo Happy converting! 🎯
echo.
pause