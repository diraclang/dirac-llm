#!/bin/bash

# Train from base model with strong signal differentiation
# 82.5% DIRAC / 17.5% general knowledge
# NO <|eot_id|> tokens
# NO <dirac> wrapper tags - just the inner DIRAC tags

# Copy dataset without wrappers to main train.jsonl
cp dataset/dirac_data_extended/train_messages.jsonl dataset/dirac_data_extended/train.jsonl
for file in `ls ~/.dirac/training/*.jsonl`
do
 cat $file >> dataset/dirac_data_extended/train.jsonl
done
cat dataset/dirac_data_extended/bash_commands_training.jsonl >> dataset/dirac_data_extended/train.jsonl   

mlx_lm_lora.train \
--model mlx-community/Mistral-7B-Instruct-v0.3 \
--train \
--data dataset/dirac_data_extended \
--batch-size 1 \
--iters 600 \
--adapter-path llm_models/mistral_clean

echo ""
echo "Training complete!"
echo "New model saved to: llm_models/mistral_clean"
echo ""
echo "Dataset composition:"
echo "  - DIRAC 'In diraclang' examples: 502"
echo "  - DIRAC explicit requests: 20"
echo "  - General knowledge: 111"
echo "  - Total: 633 examples"
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
