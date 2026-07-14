# LLM Training Project

This project provides tools and scripts for fine-tuning Large Language Models (LLMs) using Apple's MLX framework on macOS.

## Features

- 🚀 Easy setup with automated environment configuration
- 🔧 Configurable training parameters via YAML
- 📦 LoRA fine-tuning support for efficient training
- 🎯 Multiple dataset management
- 🐍 Python wrapper for streamlined training
- 📊 Support for various MLX-compatible models

## Quick Start

### 1. Setup Environment

Run the automated setup script:

```bash
bash setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Set up the necessary directory structure

### 2. Activate Environment

```bash
source .venv/bin/activate
```

### 3. Prepare Your Dataset

Place your training data in `mlx/dataset/`. The data should be in JSONL format with chat-style conversations.

### 4. Run Training

**Option A: Using Python wrapper (Recommended)**

```bash
cd mlx
python python_script/train.py --dataset extended --iters 600
```

**Option B: Using shell scripts**

```bash
cd mlx
bash lora_train_mistral.sh
```

## Configuration

Edit `mlx/config.yml` to customize:

- Model selection (Mistral, Llama, etc.)
- Dataset paths
- Training parameters (batch size, iterations, learning rate)
- Output directories

## Directory Structure

```
llm/
├── setup.sh                 # Automated setup script
├── requirements.txt         # Python dependencies
├── mlx/
│   ├── config.yml          # Main configuration file
│   ├── dataset/            # Training datasets
│   ├── llm_models/         # Saved fused runnable models
│   ├── python_script/      # Python utilities
│   │   ├── train.py       # Training wrapper
│   │   └── download_model.py  # Model downloader
│   └── *.sh               # Training shell scripts
└── README.md

```

## Training Commands

### Basic Training
```bash
python python_script/train.py --dataset extended --iters 600
```

By default this now does two things:
- Trains using temporary internal LoRA adapters
- Saves a fused full runnable model at `mlx/llm_models/model_complete`

### Custom Parameters
```bash
python python_script/train.py \
  --dataset complete \
  --iters 1000 \
  --batch-size 2 \
  --output llm_models/my_custom_model
```

## Downloading Models

List available models:
```bash
python python_script/download_model.py --list
```

Download a specific model:
```bash
python python_script/download_model.py --model mlx-community/Mistral-7B-Instruct-v0.3
```

## Distribution

To distribute this project to others:

1. **Include these essential files:**
   - `setup.sh`
   - `requirements.txt`
   - `mlx/config.yml`
   - `mlx/python_script/*`
   - Training scripts (`mlx/*.sh`)
   - Documentation files

2. **Share datasets separately** (or provide download instructions)

3. **Recipients should:**
   - Clone/copy the repository
   - Run `bash setup.sh`
   - Activate the environment
   - Place datasets in `mlx/dataset/`
   - Run training

## Requirements

- macOS with Apple Silicon (M1/M2/M3 or later)
- Python 3.9+
- ~10GB free disk space (for models and datasets)

## For More Information

- [MLX Setup Guide](README_MLX_SETUP.md)
- [Dataset Training Guide](mlx/dataset/README_TRAINING_GENERATOR.md)

## Intention

This is a testing project for LLM fine-tuning, with the intention to run and train LLMs locally using Python on macOS.
