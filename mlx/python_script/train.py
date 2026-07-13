#!/usr/bin/env python3
"""
MLX Training Runner
A Python wrapper for running MLX LoRA training with configuration management
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path


def load_config(config_path: str = "config.yml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_absolute_path(base_dir: str, relative_path: str) -> str:
    """Convert relative path to absolute path based on base directory"""
    return str(Path(base_dir) / relative_path)


def run_training(
    config: dict,
    dataset: str = "complete",
    iters: int = None,
    batch_size: int = None,
    resume_from: str = None,
    output_path: str = None
):
    """Run MLX LoRA training with specified configuration"""
    
    # Get base directory (mlx/ directory, one level up from python_script/)
    base_dir = Path(__file__).parent.parent
    
    # Build dataset path
    dataset_dir = config.get('dataset_dir', 'dataset')
    datasets = config.get('datasets', {})
    dataset_path = datasets.get(dataset, dataset)
    full_dataset_path = base_dir / dataset_dir / dataset_path
    
    if not full_dataset_path.exists():
        print(f"Error: Dataset not found at {full_dataset_path}")
        sys.exit(1)
    
    # Get model name
    model_name = config.get('model', {}).get('name', 'mlx-community/Mistral-7B-Instruct-v0.3')
    
    # Get training parameters
    training_config = config.get('training', {})
    batch_size = batch_size or training_config.get('batch_size', 1)
    iters = iters or training_config.get('iters', 600)
    
    # Build command
    cmd = [
        'mlx_lm_lora.train',
        '--model', model_name,
        '--train',
        '--data', str(full_dataset_path),
        '--batch-size', str(batch_size),
        '--iters', str(iters)
    ]
    
    # Add resume option if specified
    if resume_from:
        resume_path = base_dir / resume_from
        if resume_path.exists():
            cmd.extend(['--resume-adapter-file', str(resume_path)])
        else:
            print(f"Warning: Resume file not found at {resume_path}")
    
    # Add output path if specified
    if output_path:
        output_full_path = base_dir / output_path
        cmd.extend(['--adapter-path', str(output_full_path)])
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Dataset: {full_dataset_path}")
    print(f"Model: {model_name}")
    print(f"Iterations: {iters}, Batch size: {batch_size}")
    print("-" * 60)
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(description='MLX LoRA Training Runner')
    parser.add_argument(
        '--config',
        default='config.yml',
        help='Path to configuration file (default: config.yml)'
    )
    parser.add_argument(
        '--dataset',
        default='complete',
        help='Dataset to use (from config.yml datasets section)'
    )
    parser.add_argument(
        '--iters',
        type=int,
        help='Number of training iterations (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--resume',
        help='Path to adapter file to resume from (e.g., llm_models/mistral_finetuned/adapters.safetensors)'
    )
    parser.add_argument(
        '--output',
        help='Output path for trained adapters (e.g., llm_models/my_model)'
    )
    
    args = parser.parse_args()
    
    # Change to mlx directory (one level up from python_script/)
    mlx_dir = Path(__file__).parent.parent
    os.chdir(mlx_dir)
    
    # Load configuration (relative to mlx directory)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = mlx_dir / config_path
    
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Run training
    run_training(
        config=config,
        dataset=args.dataset,
        iters=args.iters,
        batch_size=args.batch_size,
        resume_from=args.resume,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
