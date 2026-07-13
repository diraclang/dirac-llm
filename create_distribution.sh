#!/bin/bash

# Package Distribution Script
# Creates a distribution-ready archive of the MLX training project

set -e

echo "========================================="
echo "MLX Training Distribution Packager"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Package name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="mlx-training-${TIMESTAMP}"
OUTPUT_FILE="${PACKAGE_NAME}.tar.gz"

echo -e "${BLUE}Creating distribution package: ${OUTPUT_FILE}${NC}"
echo ""

# Ask what to include
echo "What would you like to include?"
echo "1) Essential files only (recommended, ~100KB-1MB)"
echo "2) Essential + dataset generators (~1-5MB)"
echo "3) Essential + datasets (~50-500MB depending on datasets)"
echo "4) Everything including models (very large, >5GB)"
echo ""
read -p "Select option (1-4): " -n 1 -r
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case $REPLY in
    1)
        echo -e "${BLUE}Packaging essential files only...${NC}"
        tar -czf "$OUTPUT_FILE" \
            --exclude='.venv' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.DS_Store' \
            --exclude='mlx/llm_models' \
            --exclude='mlx/adapters/*.safetensors' \
            --exclude='mlx/dataset/*/*.jsonl' \
            --exclude='mlx/dataset/*.jsonl' \
            --exclude='mlx/mlx-lm-lora' \
            -C "$(dirname "$SCRIPT_DIR")" \
            "$(basename "$SCRIPT_DIR")/setup.sh" \
            "$(basename "$SCRIPT_DIR")/requirements.txt" \
            "$(basename "$SCRIPT_DIR")/README.md" \
            "$(basename "$SCRIPT_DIR")/README_MLX_SETUP.md" \
            "$(basename "$SCRIPT_DIR")/DISTRIBUTION.md" \
            "$(basename "$SCRIPT_DIR")/.gitignore" \
            "$(basename "$SCRIPT_DIR")/mlx/config.yml" \
            "$(basename "$SCRIPT_DIR")/mlx/python_script/" \
            "$(basename "$SCRIPT_DIR")/mlx/"*.sh
        ;;
    2)
        echo -e "${BLUE}Packaging essential files + dataset generators...${NC}"
        tar -czf "$OUTPUT_FILE" \
            --exclude='.venv' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.DS_Store' \
            --exclude='mlx/llm_models' \
            --exclude='mlx/adapters/*.safetensors' \
            --exclude='mlx/dataset/*/*.jsonl' \
            --exclude='mlx/dataset/*.jsonl' \
            --exclude='mlx/mlx-lm-lora' \
            -C "$(dirname "$SCRIPT_DIR")" \
            "$(basename "$SCRIPT_DIR")/setup.sh" \
            "$(basename "$SCRIPT_DIR")/requirements.txt" \
            "$(basename "$SCRIPT_DIR")/README.md" \
            "$(basename "$SCRIPT_DIR")/README_MLX_SETUP.md" \
            "$(basename "$SCRIPT_DIR")/DISTRIBUTION.md" \
            "$(basename "$SCRIPT_DIR")/.gitignore" \
            "$(basename "$SCRIPT_DIR")/mlx/"
        ;;
    3)
        echo -e "${BLUE}Packaging with datasets...${NC}"
        tar -czf "$OUTPUT_FILE" \
            --exclude='.venv' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.DS_Store' \
            --exclude='mlx/llm_models' \
            --exclude='mlx/adapters/*.safetensors' \
            --exclude='mlx/mlx-lm-lora' \
            -C "$(dirname "$SCRIPT_DIR")" \
            "$(basename "$SCRIPT_DIR")/"
        ;;
    4)
        echo -e "${YELLOW}Warning: This will create a very large file!${NC}"
        read -p "Continue? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
        echo -e "${BLUE}Packaging everything...${NC}"
        tar -czf "$OUTPUT_FILE" \
            --exclude='.venv' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.DS_Store' \
            -C "$(dirname "$SCRIPT_DIR")" \
            "$(basename "$SCRIPT_DIR")/"
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

# Get file size
FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

echo ""
echo "========================================="
echo -e "${GREEN}✓ Package created successfully!${NC}"
echo "========================================="
echo ""
echo "File: $OUTPUT_FILE"
echo "Size: $FILE_SIZE"
echo ""
echo "To distribute:"
echo "  1. Share this file with recipients"
echo "  2. Recipients extract with: tar -xzf $OUTPUT_FILE"
echo "  3. Recipients run: cd llm && bash setup.sh"
echo ""
echo "For git distribution, see DISTRIBUTION.md"
echo ""
