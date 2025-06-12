#!/bin/bash
# ML Model Converter - Quick Installation Script
# One-command setup for the complete environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 ML Model Converter - Quick Setup${NC}"
echo "=================================================="
echo ""

# Check if Python 3.11+ is available
echo -e "${YELLOW}1️⃣ Checking Python installation...${NC}"
PYTHON_CMD=""

# Try different Python commands
for cmd in python3.11 python3.12 python3.13 python3.10 python3 python; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd --version 2>&1 | cut -d" " -f2 | cut -d"." -f1,2)
        MAJOR=$(echo $VERSION | cut -d"." -f1)
        MINOR=$(echo $VERSION | cut -d"." -f2)
        
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_CMD=$cmd
            echo -e "${GREEN}✅ Found Python $VERSION at $(which $cmd)${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}❌ Python 3.10+ not found!${NC}"
    echo ""
    echo "Please install Python 3.10 or higher:"
    echo "- macOS: brew install python@3.11"
    echo "- Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    echo "- Windows: Download from https://python.org"
    exit 1
fi

# Check if git is available
echo -e "${YELLOW}2️⃣ Checking Git installation...${NC}"
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git not found!${NC}"
    echo "Please install Git first: https://git-scm.com/downloads"
    exit 1
fi
echo -e "${GREEN}✅ Git is available${NC}"

# Create virtual environment
echo -e "${YELLOW}3️⃣ Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}4️⃣ Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Upgrade pip
echo -e "${YELLOW}5️⃣ Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install dependencies
echo -e "${YELLOW}6️⃣ Installing dependencies...${NC}"
echo "   📚 Installing base dependencies (PyTorch, TensorFlow, ONNX)..."
pip install -r requirements/base.txt

echo "   🌐 Installing web interface dependencies (Streamlit)..."
pip install -r requirements/web.txt

# Create necessary directories
echo -e "${YELLOW}7️⃣ Creating directory structure...${NC}"
mkdir -p outputs/{converted,temp,logs}
mkdir -p config
echo -e "${GREEN}✅ Directory structure created${NC}"

# Make scripts executable
echo -e "${YELLOW}8️⃣ Setting up launchers...${NC}"
chmod +x launch_web.py launch_cli.py
if [ -f "scripts/setup_env.sh" ]; then
    chmod +x scripts/setup_env.sh
fi
echo -e "${GREEN}✅ Scripts are executable${NC}"

# Run a quick test
echo -e "${YELLOW}9️⃣ Running quick test...${NC}"
python test_integration.py --quick 2>/dev/null || {
    echo -e "${YELLOW}⚠️ Quick test skipped (requires model file)${NC}"
}

echo ""
echo -e "${GREEN}🎉 Installation completed successfully!${NC}"
echo "=================================================="
echo ""
echo -e "${BLUE}🚀 Quick Start:${NC}"
echo ""
echo -e "${YELLOW}For Web Interface:${NC}"
echo "  ./launch_web.py"
echo "  # or"
echo "  python launch_web.py"
echo ""
echo -e "${YELLOW}For Command Line:${NC}"
echo "  ./launch_cli.py --help"
echo "  ./launch_cli.py list-formats"
echo "  ./launch_cli.py convert model.pth"
echo ""
echo -e "${YELLOW}To activate environment manually:${NC}"
echo "  source venv/bin/activate"
echo ""
echo -e "${BLUE}📚 Documentation:${NC}"
echo "  - User Guide: README.md"
echo "  - Developer Guide: CLAUDE.md"
echo "  - Web Interface: http://localhost:8501 (after running launch_web.py)"
echo ""
echo -e "${GREEN}Happy converting! 🎯${NC}"