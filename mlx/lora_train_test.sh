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
for file in `ls ~/.dirac/training/*.jsonl`
do
 cat $file >> dataset/dirac_data_extended/train.jsonl
done
cat dataset/dirac_data_extended/bash_commands_training.jsonl >> dataset/dirac_data_extended/train.jsonl   

#