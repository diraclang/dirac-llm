#!/bin/bash

# Continue training Mistral model with balanced dataset
# This resumes from previous adapters and trains on DIRAC + general knowledge
# Dataset includes 502 DIRAC examples + 40 general knowledge examples = 542 total

# Copy balanced dataset to main train.jsonl
cp dataset/dirac_data_extended/train_balanced.jsonl dataset/dirac_data_extended/train.jsonl

mlx_lm_lora.train \
--model mlx-community/Mistral-7B-Instruct-v0.3 \
--train \
--data dataset/dirac_data_extended \
--batch-size 1 \
--iters 400 \
--resume-adapter-file llm_models/mistral_finetuned/adapters.safetensors \
--adapter-path llm_models/mistral_finetuned_balanced
