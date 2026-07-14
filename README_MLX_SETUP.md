# MLX, MLX-LM, and MLX-LM-LORA Installation & Usage Guide

## 1. Install MLX

MLX is Apple's machine learning framework for macOS. To install:

### Requirements
- macOS (Apple Silicon recommended)
- Python 3.9+

### Installation
```zsh
pip install mlx
```

## 2. Install MLX-LM

MLX-LM is a language model library built on MLX.

### Installation
```zsh
pip install mlx-lm
```


## 3. Install MLX-LM-LORA (Recommended)

MLX-LM-LORA is a LoRA fine-tuning library for MLX-LM. The easiest way to get started is to install it via pip:

### Create and activate a virtual environment (recommended)
```zsh
python3 -m venv .venv
source .venv/bin/activate
```

### Install MLX-LM-LORA
```zsh
pip install mlx-lm-lora
```

## 4. Run MLX-LM-LORA Training

### Example command
```zsh
python -m mlx_lm_lora.train --model mlx-community/Mistral-7B-Instruct-v0.3 --train --data dataset/dirac_data_2 --iters 600
```


Or use the provided shell script:
```zsh
./script/lora_train_llama.sh
```

## Dataset Format

The `dataset/dirac_data_2` directory contains the training data for fine-tuning. It should include `train.jsonl`, and may also include `valid.jsonl` and `test.jsonl`.

### `train.jsonl` Format
`train.jsonl` is a JSONL (JSON Lines) file. Each line is a single JSON object representing one training entry. Entries are separated by newlines (not commas).

If `valid.jsonl` is missing, `mlx/python_script/train.py` will automatically create a temporary validation split from the last 10% of `train.jsonl`. You can override that ratio with `training.validation_split` in `mlx/config.yml`.

Format:
```
{"prompt": "<your prompt text>", "completion": "<expected completion text>"}
{"prompt": "What is the capital of France?", "completion": "Paris."}
```

Each line must be a valid JSON object. Do not use commas between entries—only newlines separate them.

This format is used for supervised fine-tuning, where the model learns to map prompts to completions.

## Advanced: Clone and Install from Source

For advanced users or developers, you can clone the repository and install dependencies manually:
```zsh
git clone https://github.com/ml-examples/mlx-lm-lora.git
cd mlx-lm-lora
pip install -r requirements.txt
```

This allows you to modify the source code or use unreleased features.

## 5. Troubleshooting
- Ensure your virtual environment is activated before running training.
- If you encounter memory errors, reduce the batch size or sequence length.
- Make sure all dependencies are installed (`pip install -r requirements.txt`).

## 6. References
- [MLX GitHub](https://github.com/apple/mlx)
- [MLX-LM GitHub](https://github.com/ml-examples/mlx-lm)
- [MLX-LM-LORA GitHub](https://github.com/ml-examples/mlx-lm-lora)

---
This guide summarizes the installation and usage steps for MLX, MLX-LM, and MLX-LM-LORA. Adjust paths and commands as needed for your environment.
