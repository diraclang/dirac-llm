#!/bin/bash

# MLX Training Setup Script
# This script sets up the environment for MLX model training and distribution

set -e  # Exit on error

echo "========================================="
echo "MLX Training Environment Setup"
echo "========================================="
echo ""

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${YELLOW}Warning: MLX is optimized for macOS with Apple Silicon.${NC}"
    echo "You may experience issues on other platforms."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo -e "${YELLOW}Python 3.9+ is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Create virtual environment
echo ""
echo -e "${BLUE}Creating virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo -e "${BLUE}Activating virtual environment...${NC}"
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install requirements
echo ""
echo -e "${BLUE}Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create necessary directories
echo ""
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p mlx/dataset
mkdir -p mlx/llm_models
mkdir -p mlx/adapters
mkdir -p mlx/script
mkdir -p mlx/python_script
echo -e "${GREEN}✓ Directories created${NC}"

# Check if config.yml exists
echo ""
if [ ! -f "mlx/config.yml" ]; then
    echo -e "${YELLOW}Warning: mlx/config.yml not found${NC}"
    echo "Please ensure the config.yml file is present in mlx/ directory"
else
    echo -e "${GREEN}✓ Configuration file found${NC}"
fi

# Summary
echo ""
echo "========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     ${BLUE}source .venv/bin/activate${NC}"
echo ""
echo "  2. Place your training datasets in mlx/dataset/"
echo ""
echo "  3. Run a training script:"
echo "     ${BLUE}cd mlx && bash lora_train_mistral.sh${NC}"
echo ""
echo "  4. Or use the Python training wrapper:"
echo "     ${BLUE}python mlx/python_script/train.py${NC}"
echo ""
echo "For more information, see README.md"
echo ""
