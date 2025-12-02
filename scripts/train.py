#!/usr/bin/env python3
"""
Train design-analyst LoRA adapter using MLX.

Usage:
    python scripts/train.py
    python scripts/train.py --config lora_config.yaml
    python scripts/train.py --resume adapters/design-analyst-v4

Outputs:
    adapters/design-analyst-v{N}/
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_next_version(base_path: str = "adapters") -> int:
    """Find next version number."""
    adapters_dir = Path(base_path)
    if not adapters_dir.exists():
        return 1

    versions = []
    for d in adapters_dir.iterdir():
        if d.is_dir() and d.name.startswith("design-analyst-v"):
            try:
                v = int(d.name.split("-v")[-1])
                versions.append(v)
            except ValueError:
                pass

    return max(versions, default=0) + 1


def train_mlx(config_path: str = None, resume_from: str = None, output_path: str = None):
    """Train using MLX-LM."""

    if output_path is None:
        version = get_next_version()
        output_path = f"adapters/design-analyst-v{version}"

    print(f"Training output: {output_path}")

    cmd = ["python", "-m", "mlx_lm", "lora"]

    if config_path and Path(config_path).exists():
        cmd.extend(["--config", config_path])
    else:
        # Default config
        cmd.extend([
            "--model", "microsoft/Phi-3-mini-4k-instruct",
            "--train",
            "--data", "data/mlx_final",
            "--batch-size", "4",
            "--iters", "600",
            "--learning-rate", "1e-5",
            "--num-layers", "16",
            "--adapter-path", output_path,
        ])

    if resume_from:
        cmd.extend(["--resume-adapter-file", f"{resume_from}/adapters.safetensors"])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✅ Training complete: {output_path}")
    else:
        print(f"\n❌ Training failed")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train design-analyst LoRA")
    parser.add_argument("--config", "-c", default="lora_config.yaml",
                        help="Training config YAML")
    parser.add_argument("--resume", "-r",
                        help="Resume from adapter path")
    parser.add_argument("--output", "-o",
                        help="Output adapter path")
    parser.add_argument("--data", "-d", default="data/mlx_final",
                        help="Training data directory")
    args = parser.parse_args()

    train_mlx(args.config, args.resume, args.output)


if __name__ == "__main__":
    main()