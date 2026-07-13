# Distribution Guide

## How to Distribute This MLX Training Project

This guide explains how to package and share this project with others.

---

## What to Include

### Essential Files (Always Include)

```
llm/
├── setup.sh                     # ✅ Main setup script
├── requirements.txt             # ✅ Python dependencies
├── README.md                    # ✅ Main documentation
├── README_MLX_SETUP.md         # ✅ MLX-specific setup guide
└── mlx/
    ├── config.yml              # ✅ Configuration template
    ├── python_script/          # ✅ All Python utilities
    │   ├── train.py
    │   └── download_model.py
    ├── lora_train_*.sh         # ✅ Training scripts
    └── dataset/
        └── README_TRAINING_GENERATOR.md  # ✅ Dataset guide
```

### Optional Files (Include if Available)

```
mlx/
├── dataset/                    # ⚠️  Large - consider separate distribution
│   ├── *.py                   # Include dataset generators
│   └── */train.jsonl          # Datasets (can be large)
├── llm_models/                 # ⚠️  Very large - usually excluded
└── adapters/                   # ⚠️  Large - usually excluded
```

---

## Distribution Methods

### Method 1: Git Repository (Recommended)

**Best for:** Team collaboration, version control

```bash
# Create a distribution repository
cd /Users/zhiwang/llm

# Initialize git (if not already)
git init

# Create .gitignore to exclude large files
cat > .gitignore << EOF
# Virtual environment
.venv/
venv/
__pycache__/

# Large model files
mlx/llm_models/
mlx/adapters/*.safetensors

# Large datasets (optional - comment out if you want to include)
mlx/dataset/*/train.jsonl
mlx/dataset/*/test.jsonl

# OS files
.DS_Store
EOF

# Commit essential files
git add setup.sh requirements.txt README.md
git add mlx/config.yml mlx/python_script/ mlx/*.sh
git commit -m "Initial distribution package"

# Push to remote (GitHub, GitLab, etc.)
git remote add origin <your-repo-url>
git push -u origin main
```

**Recipients then:**
```bash
git clone <your-repo-url>
cd llm
bash setup.sh
```

---

### Method 2: Compressed Archive

**Best for:** One-time distribution, offline sharing

```bash
cd /Users/zhiwang

# Create distribution archive (excluding large files)
tar -czf llm-training-dist.tar.gz \
  --exclude='llm/.venv' \
  --exclude='llm/mlx/llm_models' \
  --exclude='llm/mlx/adapters/*.safetensors' \
  --exclude='llm/mlx/dataset/*/train.jsonl' \
  llm/setup.sh \
  llm/requirements.txt \
  llm/README*.md \
  llm/mlx/config.yml \
  llm/mlx/python_script/ \
  llm/mlx/*.sh \
  llm/mlx/dataset/*.py \
  llm/mlx/dataset/*.md

# Or include everything (very large):
tar -czf llm-training-full.tar.gz \
  --exclude='llm/.venv' \
  llm/
```

**Recipients then:**
```bash
tar -xzf llm-training-dist.tar.gz
cd llm
bash setup.sh
```

---

### Method 3: Docker Container (Advanced)

**Best for:** Reproducible environments, complex dependencies

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy project files
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY mlx/ mlx/
COPY setup.sh README.md ./

CMD ["/bin/bash"]
```

**Build and distribute:**
```bash
docker build -t mlx-training .
docker save mlx-training | gzip > mlx-training-docker.tar.gz
```

---

## Separate Dataset Distribution

Since datasets can be large, consider distributing them separately:

### Option A: Cloud Storage

```bash
# Upload to cloud (S3, Google Drive, Dropbox, etc.)
# Then provide download instructions in README

echo "Download dataset from: https://your-storage.com/datasets.zip" >> DATASET_DOWNLOAD.md
```

### Option B: Dataset Generator Scripts

Include scripts to generate/download datasets:

```bash
# Already included in mlx/dataset/*.py
# Recipients can run these to generate data
cd mlx/dataset
python generate_training_from_index.js
```

---

## Distribution Checklist

Before distributing, verify:

- [ ] `setup.sh` runs successfully on a fresh machine
- [ ] `requirements.txt` includes all dependencies
- [ ] `config.yml` has sensible defaults (no hardcoded absolute paths)
- [ ] Documentation includes:
  - [ ] Installation instructions
  - [ ] Training examples
  - [ ] System requirements
  - [ ] Troubleshooting section
- [ ] No sensitive information (API keys, tokens, etc.)
- [ ] No unnecessary large files (>100MB)
- [ ] Scripts use relative paths (not absolute)

---

## Example Distribution Package Structure

```
llm-training-package/
├── README.md                    # Start here
├── DISTRIBUTION.md              # This file
├── setup.sh                     # Run this first
├── requirements.txt             # Dependencies
├── mlx/
│   ├── config.yml              # Configuration
│   ├── python_script/          # Utilities
│   ├── lora_train_*.sh         # Training scripts
│   └── dataset/
│       ├── README_TRAINING_GENERATOR.md
│       └── *.py                # Dataset generators
└── docs/
    └── README_MLX_SETUP.md     # Detailed setup guide
```

---

## Recipient Quick Start

After receiving the distribution:

```bash
# 1. Extract/clone the project
cd llm

# 2. Run setup
bash setup.sh

# 3. Activate environment
source .venv/bin/activate

# 4. Download model (optional - will auto-download during training)
python mlx/python_script/download_model.py --model mlx-community/Mistral-7B-Instruct-v0.3

# 5. Prepare dataset
# Place your training data in mlx/dataset/

# 6. Start training
cd mlx
python python_script/train.py --dataset complete --iters 600
```

---

## Troubleshooting for Recipients

Common issues and solutions:

| Issue | Solution |
|-------|----------|
| "Python 3.9+ required" | Install Python 3.9 or higher |
| "MLX is optimized for macOS" | Use macOS with Apple Silicon, or accept warning |
| "Dataset not found" | Place datasets in `mlx/dataset/` |
| "Model download fails" | Check internet connection and HuggingFace access |
| "Permission denied" | Run `chmod +x setup.sh` |

---

## Support

For issues or questions about this distribution:
1. Check README.md for common solutions
2. Review config.yml for configuration options
3. Contact the distributor for support

---

**Last Updated:** 2026-07-13
