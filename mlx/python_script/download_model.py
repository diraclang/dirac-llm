#!/usr/bin/env python3
"""
Download MLX Models
Downloads and caches MLX models from Hugging Face Hub
"""

import argparse
import os
from pathlib import Path


def download_model(model_name: str, output_dir: str = None):
    """Download a model from Hugging Face Hub using MLX"""
    try:
        from mlx_lm import load
        
        print(f"Downloading model: {model_name}")
        print("This may take a while depending on your internet connection...")
        
        # Load the model (this will download and cache it)
        model, tokenizer = load(model_name)
        
        print(f"✓ Model {model_name} downloaded and cached successfully!")
        print(f"  Cache location: ~/.cache/huggingface/hub/")
        
        if output_dir:
            print(f"  Note: MLX uses Hugging Face cache. To save elsewhere, use HF tools.")
        
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


def list_common_models():
    """List commonly used MLX models"""
    models = [
        "mlx-community/Mistral-7B-Instruct-v0.3",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "mlx-community/gemma-2-9b-it-4bit",
    ]
    
    print("Common MLX models:")
    print("-" * 60)
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='Download MLX models from Hugging Face')
    parser.add_argument(
        '--model',
        help='Model name to download (e.g., mlx-community/Mistral-7B-Instruct-v0.3)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List common MLX models'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_common_models()
        return
    
    if not args.model:
        print("Error: Please specify a model name with --model or use --list to see options")
        print("\nExample:")
        print("  python download_model.py --model mlx-community/Mistral-7B-Instruct-v0.3")
        return
    
    download_model(args.model)


if __name__ == '__main__':
    main()
