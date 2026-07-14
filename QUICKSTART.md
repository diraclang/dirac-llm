# Quick Start Guide for Recipients

Welcome! This guide will help you get started with the MLX training project quickly.

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4 or later recommended)
- **Python 3.9+** installed
- **~10GB free disk space** (for models and datasets)
- **Internet connection** (for downloading models)

---

## Setup (5 minutes)

### 1. Extract the Project

If you received a `.tar.gz` file:
```bash
tar -xzf mlx-training-*.tar.gz
cd llm
```

If you cloned from git:
```bash
git clone <repository-url>
cd llm
```

### 2. Run Setup Script

```bash
bash setup.sh
```

This automatically:
- ✅ Checks Python version
- ✅ Creates virtual environment
- ✅ Installs all dependencies
- ✅ Creates necessary directories

### 3. Activate Environment

```bash
source .venv/bin/activate
```

**Important:** You must activate the environment every time you open a new terminal!

---

## Training Your First Model (10 minutes)

### Option A: Quick Test (Recommended for First Time)

```bash
cd mlx

# Train with default settings (600 iterations)
python python_script/train.py --dataset complete --iters 100
```

This will:
1. Auto-download the Mistral-7B model (~4GB)
2. Load your training dataset
3. Start LoRA fine-tuning
4. Save a full runnable model in `mlx/llm_models/model_complete` (default)

### Option B: Full Training

```bash
cd mlx
python python_script/train.py --dataset complete --iters 600
```

---

## Adding Your Own Data

### 1. Prepare Your Dataset

Your training data should be in JSONL format (one JSON object per line):

```jsonl
{"messages": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is a programming language..."}]}
{"messages": [{"role": "user", "content": "How do I create a list?"}, {"role": "assistant", "content": "Use square brackets..."}]}
```

### 2. Place Dataset

```bash
mkdir -p mlx/dataset/my_data
# Copy your train.jsonl file into mlx/dataset/my_data/
```

If your dataset folder does not include a `valid.jsonl`, the training wrapper will automatically carve off the last 10% of `train.jsonl` as validation data. You can tune that ratio with `training.validation_split` in `mlx/config.yml` or provide your own `valid.jsonl` explicitly.

### 3. Update Configuration

Edit `mlx/config.yml`:

```yaml
datasets:
  my_data: "my_data"  # Add this line
```

### 4. Train

```bash
python python_script/train.py --dataset my_data --iters 600
```

---

## Common Commands

### List Available Models
```bash
python mlx/python_script/download_model.py --list
```

### Download Specific Model
```bash
python mlx/python_script/download_model.py --model mlx-community/Mistral-7B-Instruct-v0.3
```

### Change Training Parameters
```bash
python mlx/python_script/train.py \
  --dataset complete \
  --iters 1000 \
  --batch-size 2
```

---

## What's in the Config File?

Edit `mlx/config.yml` to customize:

```yaml
# Model to use
model:
  name: "mlx-community/Mistral-7B-Instruct-v0.3"

# Training parameters
training:
  batch_size: 1        # Increase if you have enough RAM
  iters: 600          # Number of training iterations
  learning_rate: 1e-5

# Your datasets
datasets:
  complete: "dirac_data_complete"
  my_data: "my_custom_data"
```

---

## Troubleshooting

### "Python 3.9+ required"
**Solution:** Install Python 3.9 or higher
```bash
brew install python@3.11
```

### "MLX is optimized for macOS"
**Solution:** This project works best on Apple Silicon Macs. You can continue on Intel Macs or Linux, but performance will be limited.

### "Dataset not found"
**Solution:** Make sure your dataset folder exists in `mlx/dataset/` and contains a `train.jsonl` file.

### "Validation set not found or empty"
**Solution:** Add a `valid.jsonl` next to `train.jsonl`, or use the Python training wrapper which now auto-generates a temporary validation split from `train.jsonl`.

### "Model download fails"
**Solution:** Check your internet connection. Models download from Hugging Face and can be 4-8GB.

### "Out of memory"
**Solution:** Reduce batch size in config.yml or use a smaller model.

### Permission denied on setup.sh
**Solution:** 
```bash
chmod +x setup.sh
bash setup.sh
```

---

## Next Steps

1. **Read the full documentation:** `README.md`
2. **Learn about MLX:** `README_MLX_SETUP.md`
3. **Understand distribution:** `DISTRIBUTION.md`
4. **Check dataset tools:** `mlx/dataset/README_TRAINING_GENERATOR.md`

---

## Project Structure

```
llm/
├── setup.sh              ← Run this first
├── requirements.txt      ← Python dependencies
├── README.md             ← Full documentation
├── QUICKSTART.md         ← You are here!
├── mlx/
│   ├── config.yml       ← Configure training here
│   ├── dataset/         ← Put your data here
│   ├── llm_models/      ← Trained models saved here
│   ├── python_script/   ← Training utilities
│   │   ├── train.py    ← Main training script
│   │   └── download_model.py
│   └── *.sh            ← Alternative shell scripts
```

---

## Getting Help

1. Check error messages carefully - they often explain the issue
2. Review the configuration in `mlx/config.yml`
3. Make sure you've activated the virtual environment
4. Check that datasets are in the correct format
5. Contact the person who shared this project with you

---

## Example Training Session

Complete example from start to finish:

```bash
# 1. Setup (one time only)
bash setup.sh
source .venv/bin/activate

# 2. Add your dataset
mkdir -p mlx/dataset/my_training_data
# Copy your train.jsonl to mlx/dataset/my_training_data/

# 3. Edit config
# Add "my_training_data" to datasets in mlx/config.yml

# 4. Start training
cd mlx
python python_script/train.py \
  --dataset my_training_data \
  --iters 600 \
  --output llm_models/my_first_model

# 5. Models saved to: mlx/llm_models/my_first_model/
```

---

**Ready to begin?** Run `bash setup.sh` and follow the prompts!

**Last Updated:** 2026-07-13
