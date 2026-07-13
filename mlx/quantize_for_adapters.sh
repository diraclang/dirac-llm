#!/bin/bash
# Quantize the base model that you're using with your adapters

source /Users/zhiwang/llm/.venv/bin/activate
cd /Users/zhiwang/llm/mlx/llm_models

# If your base model is in mlx_model_16, quantize it:
echo "Quantizing base model to 4-bit..."
python -m mlx_lm.convert \
    --hf-path mistral_finetuned \
    --mlx-path mistral_finetuned_4bit \
    --quantize \
    --q-bits 4

echo "Done! Now update your HTTP server to use the quantized model:"
echo "  model, tokenizer = load('../llm_models/mlx_model_16_4bit', adapter_path='../adapters')"
