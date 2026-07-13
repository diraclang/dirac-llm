#!/bin/bash

# Distribution Verification Test
# Tests that the distribution package is ready for sharing

set -e

echo "========================================="
echo "Distribution Verification Test"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 (missing)"
        ((ERRORS++))
        return 1
    fi
}

# Function to check directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
        return 0
    else
        echo -e "${RED}✗${NC} $1/ (missing)"
        ((ERRORS++))
        return 1
    fi
}

# Function to check if file is executable
check_executable() {
    if [ -x "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 (executable)"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $1 (not executable)"
        ((WARNINGS++))
        return 1
    fi
}

echo -e "${BLUE}Checking essential files...${NC}"
echo ""

# Core files
check_file "setup.sh"
check_file "requirements.txt"
check_file "README.md"
check_file "QUICKSTART.md"
check_file "DISTRIBUTION.md"
check_file ".gitignore"
check_file "create_distribution.sh"

echo ""
echo -e "${BLUE}Checking MLX directory structure...${NC}"
echo ""

# MLX files
check_dir "mlx"
check_file "mlx/config.yml"
check_dir "mlx/python_script"
check_file "mlx/python_script/train.py"
check_file "mlx/python_script/download_model.py"

echo ""
echo -e "${BLUE}Checking training scripts...${NC}"
echo ""

# Training scripts
for script in mlx/*.sh; do
    if [ -f "$script" ]; then
        check_file "$script"
    fi
done

echo ""
echo -e "${BLUE}Checking executable permissions...${NC}"
echo ""

check_executable "setup.sh"
check_executable "create_distribution.sh"
check_executable "mlx/python_script/train.py"
check_executable "mlx/python_script/download_model.py"

echo ""
echo -e "${BLUE}Checking directory structure...${NC}"
echo ""

# Directories that should exist (or be created)
check_dir "mlx/dataset"
check_dir "mlx/llm_models"
check_dir "mlx/python_script"

echo ""
echo -e "${BLUE}Checking for hardcoded paths in config...${NC}"
echo ""

if grep -q "/Users/" mlx/config.yml 2>/dev/null; then
    echo -e "${RED}✗${NC} Found absolute paths in config.yml"
    echo "  Please use relative paths in configuration"
    ((ERRORS++))
else
    echo -e "${GREEN}✓${NC} No hardcoded absolute paths found"
fi

echo ""
echo -e "${BLUE}Checking Python syntax...${NC}"
echo ""

for pyfile in mlx/python_script/*.py; do
    if [ -f "$pyfile" ]; then
        if python3 -m py_compile "$pyfile" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} $pyfile (syntax OK)"
        else
            echo -e "${RED}✗${NC} $pyfile (syntax error)"
            ((ERRORS++))
        fi
    fi
done

echo ""
echo -e "${BLUE}Checking requirements.txt format...${NC}"
echo ""

if [ -f "requirements.txt" ]; then
    if grep -E "^[a-zA-Z0-9_-]+[>=<]" requirements.txt > /dev/null; then
        echo -e "${GREEN}✓${NC} requirements.txt format valid"
    else
        echo -e "${YELLOW}⚠${NC} requirements.txt may have formatting issues"
        ((WARNINGS++))
    fi
fi

echo ""
echo -e "${BLUE}Checking for sensitive files...${NC}"
echo ""

SENSITIVE_FILES=(
    ".token"
    ".env"
    "secrets.yml"
)

FOUND_SENSITIVE=false
for pattern in "${SENSITIVE_FILES[@]}"; do
    if [ -f "$pattern" ]; then
        # Check if it's in .gitignore
        if grep -q "^$pattern" .gitignore 2>/dev/null || grep -q "^\*${pattern##*.}" .gitignore 2>/dev/null; then
            echo -e "${YELLOW}⚠${NC} Found $pattern (but it's in .gitignore - OK)"
        else
            echo -e "${RED}✗${NC} Found sensitive file: $pattern"
            echo "  Make sure this is in .gitignore"
            ((ERRORS++))
            FOUND_SENSITIVE=true
        fi
    fi
done

if [ "$FOUND_SENSITIVE" = false ]; then
    echo -e "${GREEN}✓${NC} No unprotected sensitive files found"
fi

echo ""
echo -e "${BLUE}Checking .gitignore coverage...${NC}"
echo ""

SHOULD_IGNORE=(
    ".venv"
    "mlx/llm_models"
    "__pycache__"
    ".DS_Store"
)

for item in "${SHOULD_IGNORE[@]}"; do
    if grep -q "$item" .gitignore 2>/dev/null; then
        echo -e "${GREEN}✓${NC} .gitignore covers: $item"
    else
        echo -e "${YELLOW}⚠${NC} .gitignore missing: $item"
        ((WARNINGS++))
    fi
done

echo ""
echo -e "${BLUE}Estimating distribution size...${NC}"
echo ""

# Estimate size without large files
ESSENTIAL_SIZE=$(du -sh . 2>/dev/null | \
    grep -v ".venv" | \
    grep -v "llm_models" | \
    grep -v "__pycache__" | \
    head -n 1 | \
    cut -f1 || echo "Unknown")

echo "Estimated size (essential files): $ESSENTIAL_SIZE"

echo ""
echo "========================================="
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo "========================================="
    echo ""
    echo "Your distribution is ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Run: bash create_distribution.sh"
    echo "  2. Or commit to git and push"
    echo "  3. Share with recipients"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Passed with $WARNINGS warning(s)${NC}"
    echo "========================================="
    echo ""
    echo "Distribution is mostly ready, but review warnings above."
else
    echo -e "${RED}✗ Failed with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo "========================================="
    echo ""
    echo "Please fix the errors above before distributing."
    exit 1
fi

echo ""
