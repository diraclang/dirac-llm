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

temp_adapter_dir=$(mktemp -d)

# Allow override via environment variable for A/B deployment
if [[ -n "${MODEL_OUTPUT_DIR:-}" ]]; then
	final_model_dir="$MODEL_OUTPUT_DIR"
else
	final_model_dir="llm_models/mistral_clean"
fi

mlx_lm_lora.train \
--model mlx-community/Mistral-7B-Instruct-v0.3 \
--train \
--data dataset/dirac_data_extended \
--batch-size 1 \
--iters 600 \
--adapter-path "$temp_adapter_dir"

mlx_lm.fuse \
--model mlx-community/Mistral-7B-Instruct-v0.3 \
--adapter-path "$temp_adapter_dir" \
--save-path "$final_model_dir"

rm -rf "$temp_adapter_dir"

echo ""
echo "Training complete!"
echo "New fused model saved to: $final_model_dir"
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
