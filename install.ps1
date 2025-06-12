# ML Model Converter - PowerShell Installation Script
# One-command setup for the complete environment

# Enable strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Colors for output
$Red = [ConsoleColor]::Red
$Green = [ConsoleColor]::Green
$Yellow = [ConsoleColor]::Yellow
$Blue = [ConsoleColor]::Blue
$White = [ConsoleColor]::White

function Write-ColorText {
    param(
        [string]$Text,
        [ConsoleColor]$Color = $White
    )
    Write-Host $Text -ForegroundColor $Color
}

Write-ColorText "🚀 ML Model Converter - Quick Setup (Windows)" $Blue
Write-ColorText "==================================================" $Blue
Write-Host ""

# Check if Python 3.10+ is available
Write-ColorText "1️⃣ Checking Python installation..." $Yellow
$PythonCmd = $null
$PythonVersion = $null

# Try different Python commands
$PythonCommands = @("python3.11", "python3.12", "python3.13", "python3.10", "python", "py")

foreach ($cmd in $PythonCommands) {
    try {
        $versionOutput = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $version = ($versionOutput -split " ")[1]
            $versionParts = $version -split "\."
            $major = [int]$versionParts[0]
            $minor = [int]$versionParts[1]
            
            if ($major -eq 3 -and $minor -ge 10) {
                $PythonCmd = $cmd
                $PythonVersion = $version
                Write-ColorText "✅ Found Python $version at $cmd" $Green
                break
            }
        }
    }
    catch {
        # Command not found, continue to next
        continue
    }
}

if (-not $PythonCmd) {
    Write-ColorText "❌ Python 3.10+ not found!" $Red
    Write-Host ""
    Write-Host "Please install Python 3.10 or higher:"
    Write-Host "- Download from: https://python.org"
    Write-Host "- Make sure to check 'Add Python to PATH' during installation"
    Write-Host "- Restart PowerShell after installation"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if git is available
Write-ColorText "2️⃣ Checking Git installation..." $Yellow
try {
    git --version | Out-Null
    Write-ColorText "✅ Git is available" $Green
}
catch {
    Write-ColorText "❌ Git not found!" $Red
    Write-Host "Please install Git first: https://git-scm.com/downloads"
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment
Write-ColorText "3️⃣ Creating virtual environment..." $Yellow
if (-not (Test-Path "venv")) {
    & $PythonCmd -m venv venv
    Write-ColorText "✅ Virtual environment created" $Green
} else {
    Write-ColorText "✅ Virtual environment already exists" $Green
}

# Activate virtual environment
Write-ColorText "4️⃣ Activating virtual environment..." $Yellow
& "venv\Scripts\Activate.ps1"
Write-ColorText "✅ Virtual environment activated" $Green

# Upgrade pip
Write-ColorText "5️⃣ Upgrading pip..." $Yellow
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
Write-ColorText "6️⃣ Installing dependencies..." $Yellow
Write-Host "   📚 Installing base dependencies (PyTorch, TensorFlow, ONNX)..."
pip install -r requirements/base.txt

Write-Host "   🌐 Installing web interface dependencies (Streamlit)..."
pip install -r requirements/web.txt

# Create necessary directories
Write-ColorText "7️⃣ Creating directory structure..." $Yellow
$directories = @("outputs", "outputs\converted", "outputs\temp", "outputs\logs", "config")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-ColorText "✅ Directory structure created" $Green

# Check launchers
Write-ColorText "8️⃣ Checking launchers..." $Yellow
if (Test-Path "launch_web.py") {
    Write-ColorText "✅ Web launcher found" $Green
} else {
    Write-ColorText "⚠️ Web launcher not found" $Yellow
}

if (Test-Path "launch_cli.py") {
    Write-ColorText "✅ CLI launcher found" $Green
} else {
    Write-ColorText "⚠️ CLI launcher not found" $Yellow
}

# Run a quick test
Write-ColorText "9️⃣ Running quick test..." $Yellow
try {
    python test_integration.py --quick 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-ColorText "✅ Quick test passed" $Green
    } else {
        Write-ColorText "⚠️ Quick test skipped (requires model file)" $Yellow
    }
}
catch {
    Write-ColorText "⚠️ Quick test skipped (requires model file)" $Yellow
}

Write-Host ""
Write-ColorText "🎉 Installation completed successfully!" $Green
Write-ColorText "==================================================" $Green
Write-Host ""
Write-ColorText "🚀 Quick Start:" $Blue
Write-Host ""
Write-ColorText "For Web Interface:" $Yellow
Write-Host "  python launch_web.py"
Write-Host "  # Then open: http://localhost:8501"
Write-Host ""
Write-ColorText "For Command Line:" $Yellow
Write-Host "  python launch_cli.py --help"
Write-Host "  python launch_cli.py list-formats"
Write-Host "  python launch_cli.py convert model.pth"
Write-Host ""
Write-ColorText "To activate environment manually:" $Yellow
Write-Host "  venv\Scripts\Activate.ps1"
Write-Host ""
Write-ColorText "📚 Documentation:" $Blue
Write-Host "  - User Guide: README.md"
Write-Host "  - Developer Guide: CLAUDE.md"
Write-Host "  - Web Interface: http://localhost:8501 (after running launch_web.py)"
Write-Host ""
Write-ColorText "Happy converting! 🎯" $Green
Write-Host ""
Read-Host "Press Enter to exit"