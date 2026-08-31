#!/bin/bash
set -euo pipefail

# Ensure we're running from the mlx directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Train from base model with strong signal differentiation
# 82.5% DIRAC / 17.5% general knowledge
# NO <|eot_id|> tokens
# NO <dirac> wrapper tags - just the inner DIRAC tags

# Copy dataset without wrappers to main train.jsonl
cp dataset/dirac_data_extended/train_messages.jsonl dataset/dirac_data_extended/train.jsonl
echo "After copying train_messages.jsonl: $(wc -l < dataset/dirac_data_extended/train.jsonl) lines"

for file in `ls ~/.dirac/training/*.jsonl`
do
 cat $file >> dataset/dirac_data_extended/train.jsonl
 echo "After adding $file: $(wc -l < dataset/dirac_data_extended/train.jsonl) lines"
done

cat dataset/dirac_data_extended/bash_commands_training.jsonl >> dataset/dirac_data_extended/train.jsonl   
echo "After adding bash_commands: $(wc -l < dataset/dirac_data_extended/train.jsonl) lines"

# Convert train_balanced_15pct.jsonl (prompt/completion format) to messages format for validation
echo "Converting train_balanced_15pct.jsonl to validation set..."
python3 << 'EOF'
import json

with open('dataset/dirac_data_extended/train_balanced_15pct.jsonl', 'r') as f_in, \
     open('dataset/dirac_data_extended/valid.jsonl', 'w') as f_out:
    for line in f_in:
        data = json.loads(line)
        messages_format = {
            "messages": [
                {"role": "user", "content": data["prompt"]},
                {"role": "assistant", "content": data["completion"]}
            ]
        }
        f_out.write(json.dumps(messages_format) + '\n')
EOF
echo "Validation set created: $(wc -l < dataset/dirac_data_extended/valid.jsonl) examples"

# Base model to fine-tune. Defaults to the locally converted Qwen2.5-7B-Instruct
# MLX model. Uses an absolute path (anchored at this script's directory) since
# the script already cd's into mlx/, so a relative "mlx/Qwen2.5-7B-Instruct"
# would incorrectly resolve to mlx/mlx/Qwen2.5-7B-Instruct.
# Override with BASE_MODEL if you want to point at a different local path or
# a mlx-community hub repo.
if [[ -n "${BASE_MODEL:-}" ]]; then
	base_model="$BASE_MODEL"
else
	base_model="$SCRIPT_DIR/Qwen2.5-7B-Instruct"
fi

# Allow override via environment variable for A/B deployment.
# Keep the adapter checkpoint directly inside the model directory so it can
# be reused for resume/rollback without being deleted after training.
if [[ -n "${MODEL_OUTPUT_DIR:-}" ]]; then
	final_model_dir="$MODEL_OUTPUT_DIR"
else
	final_model_dir="llm_models/qwen"
fi

mkdir -p "$final_model_dir"
persistent_adapter_dir="${final_model_dir}/adapters"
mkdir -p "$persistent_adapter_dir"

echo "Base model: $base_model"
echo "Adapter checkpoint directory: $persistent_adapter_dir"

mlx_lm_lora.train \
--model "$base_model" \
--train \
--data dataset/dirac_data_extended \
--batch-size 1 \
--iters 600 \
--adapter-path "$persistent_adapter_dir"

fuse_succeeded=false
if mlx_lm.fuse \
--model "$base_model" \
--adapter-path "$persistent_adapter_dir" \
--save-path "$final_model_dir" \
--de-quantize; then
    fuse_succeeded=true
else
    echo ""
    echo "Warning: mlx_lm.fuse failed for Qwen."
    echo "Continuing with adapter-only deployment using:"
    echo "  Base model: $base_model"
    echo "  Adapters:   $persistent_adapter_dir"
fi

echo ""
echo "Training complete!"
echo "Adapter checkpoint kept at: $persistent_adapter_dir"
if [[ "$fuse_succeeded" == true ]]; then
    echo "New fused model saved to: $final_model_dir"
else
    echo "Fused model not available; use adapter-on-base inference for this slot."
fi
echo ""
echo "Dataset composition:"
echo "  - Training examples: $(wc -l < dataset/dirac_data_extended/train.jsonl)"
echo "  - Validation examples: $(wc -l < dataset/dirac_data_extended/valid.jsonl)"
echo "  - Total examples used: $(( $(wc -l < dataset/dirac_data_extended/train.jsonl) + $(wc -l < dataset/dirac_data_extended/valid.jsonl) ))"
echo ""
echo "Clean format:"
echo "  - NO <|eot_id|> tokens"
echo "  - NO <dirac> wrapper tags"
echo "  - Just: <output>text</output>, <loop>...</loop>, etc."
echo ""
echo "Signal keywords:"
echo "  - 'In diraclang...' → DIRAC tags"
echo "  - 'Write DIRAC...' → DIRAC tags"
echo "  - General questions → Plain answers"
