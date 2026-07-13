

#!/bin/bash
# Quantize MLX model to 4-bit or 8-bit for faster inference

# Activate virtual environment
source /Users/zhiwang/llm/.venv/bin/activate

# Navigate to model directory
cd /Users/zhiwang/llm/mlx/llm_models

# Quantize to 4-bit (fastest, ~4x smaller)
echo "Quantizing to 4-bit..."
python -m mlx_lm.convert \
    --hf-path mlx_model_16 \
    --mlx-path mlx_model_4bit \
    --quantize \
    -q-bits 4

# Or quantize to 8-bit (good balance)
echo "Quantizing to 8-bit..."
python -m mlx_lm.convert \
    --hf-path mlx_model_16 \
    --mlx-path mlx_model_8bit \
    --quantize \
    -q-bits 8

echo "Done! Models saved to:"
echo "  4-bit: mlx_model_4bit/"
echo "  8-bit: mlx_model_8bit/"
