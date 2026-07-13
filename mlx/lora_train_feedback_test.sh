#!/bin/bash

# Test training with chat format (messages field)
# Small dataset to verify format compatibility

echo "Testing chat format training..."
echo "Dataset: feedback_test (8 multi-turn examples)"
echo ""

mlx_lm_lora.train \
--model mlx-community/Mistral-7B-Instruct-v0.3 \
--train \
--data dataset/feedback_test \
--batch-size 1 \
--iters 50 \
--adapter-path llm_models/feedback_test

echo ""
echo "Training complete!"
echo "Test model saved to: llm_models/feedback_test"
echo ""
echo "Dataset details:"
echo "  - Format: chat (messages field)"
echo "  - Examples: 8 multi-turn conversations"
echo "  - Pattern: user query → assistant response → feedback → DONE/correction"
echo ""
echo "To test the model:"
echo "  cd /Users/zhiwang/llm/mlx/python_script"
echo "  python stateless_chat_server.py"
