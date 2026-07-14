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
import tempfile
import shutil
from pathlib import Path


def load_config(config_path: str = "config.yml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_absolute_path(base_dir: str, relative_path: str) -> str:
    """Convert relative path to absolute path based on base directory"""
    return str(Path(base_dir) / relative_path)


def _read_jsonl_lines(file_path: Path) -> list:
    """Read non-empty JSONL lines from a file."""
    if not file_path.exists():
        return []
    with open(file_path, 'r') as file_handle:
        return [line for line in file_handle if line.strip()]


def prepare_dataset_dir(config: dict, dataset_path: Path) -> tuple:
    """Ensure the dataset directory contains a validation split expected by mlx_lm_lora."""
    train_path = dataset_path / 'train.jsonl'
    valid_path = dataset_path / 'valid.jsonl'
    test_path = dataset_path / 'test.jsonl'

    valid_lines = _read_jsonl_lines(valid_path)
    if valid_lines:
        return dataset_path, None

    train_lines = _read_jsonl_lines(train_path)
    if len(train_lines) < 2:
        raise ValueError(
            f"Dataset at {dataset_path} needs at least 2 records in train.jsonl to auto-create valid.jsonl."
        )

    training_config = config.get('training', {})
    validation_split = float(training_config.get('validation_split', 0.1))
    validation_split = min(max(validation_split, 0.0), 0.5)

    valid_count = max(1, int(len(train_lines) * validation_split))
    valid_count = min(valid_count, len(train_lines) - 1)

    prepared_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_path.name}_prepared_"))
    with open(prepared_dir / 'train.jsonl', 'w') as train_file:
        train_file.writelines(train_lines[:-valid_count])
    with open(prepared_dir / 'valid.jsonl', 'w') as valid_file:
        valid_file.writelines(train_lines[-valid_count:])

    test_lines = _read_jsonl_lines(test_path)
    if test_lines:
        with open(prepared_dir / 'test.jsonl', 'w') as test_file:
            test_file.writelines(test_lines)

    return prepared_dir, prepared_dir


def run_training(
    config: dict,
    dataset: str = "complete",
    iters: int = None,
    batch_size: int = None,
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
    
    try:
        prepared_dataset_path, cleanup_dir = prepare_dataset_dir(config, full_dataset_path)
    except ValueError as error:
        print(f"Error: {error}")
        sys.exit(1)

    if prepared_dataset_path != full_dataset_path:
        print(
            f"Prepared validation split: {prepared_dataset_path / 'valid.jsonl'} "
            f"from {full_dataset_path / 'train.jsonl'}"
        )

    # Get model name
    model_name = config.get('model', {}).get('name', 'mlx-community/Mistral-7B-Instruct-v0.3')
    
    # Get training parameters
    training_config = config.get('training', {})
    batch_size = batch_size or training_config.get('batch_size', 1)
    iters = iters or training_config.get('iters', 600)

    output_config = config.get('output', {})
    models_dir = config.get('models_dir', 'llm_models')
    dataset_default_output = output_config.get(dataset) or output_config.get('default') or f"{models_dir}/model_{dataset}"
    temp_adapter_dir = Path(tempfile.mkdtemp(prefix=f"{dataset}_adapters_"))
    
    # Build command
    cmd = [
        'mlx_lm_lora.train',
        '--model', model_name,
        '--train',
        '--data', str(prepared_dataset_path),
        '--batch-size', str(batch_size),
        '--iters', str(iters),
        '--adapter-path', str(temp_adapter_dir)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Dataset: {prepared_dataset_path}")
    print(f"Model: {model_name}")
    print(f"Temporary adapter path: {temp_adapter_dir}")
    print(f"Iterations: {iters}, Batch size: {batch_size}")
    print("-" * 60)
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)

        fused_output_path = output_path or dataset_default_output
        if fused_output_path:
            fused_output_full_path = base_dir / fused_output_path
            fuse_cmd = [
                'mlx_lm.fuse',
                '--model', model_name,
                '--adapter-path', str(temp_adapter_dir),
                '--save-path', str(fused_output_full_path)
            ]
            print("\n" + "-" * 60)
            print(f"Fusing adapters into full model: {' '.join(fuse_cmd)}")
            subprocess.run(fuse_cmd, check=True)
            print(f"Fused model saved to: {fused_output_full_path}")
        else:
            print("\nError: no output path configured. Use --output or set output.<dataset> in config.yml")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("Temporary adapters cleaned up after fuse.")
        print("=" * 60)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        sys.exit(e.returncode)
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
        shutil.rmtree(temp_adapter_dir, ignore_errors=True)


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
        '--output',
        help='Output path for fused full model (e.g., llm_models/my_model)'
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
        output_path=args.output
    )


if __name__ == '__main__':
    main()
