#!/usr/bin/env python3
"""
Export design-analyst model: Merge LoRA + Convert to GGUF.

Usage:
    python scripts/export.py
    python scripts/export.py --adapter adapters/design-analyst-v4
    python scripts/export.py --skip-merge  # If already merged

Pipeline:
    1. Convert MLX adapter to PEFT format
    2. Merge LoRA into base Phi-3
    3. Convert to GGUF (F16)
    4. Quantize to Q4_K_M

Outputs:
    models/design-analyst-v{N}-merged/   (HuggingFace format)
    models/design-analyst-v{N}-f16.gguf  (Full precision)
    models/design-analyst-v{N}-q4.gguf   (Quantized, ~2GB)
"""

import argparse
import subprocess
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def convert_mlx_to_peft(mlx_adapter_path: str, peft_output_path: str):
    """Convert MLX LoRA adapter to HuggingFace PEFT format."""
    print(f"Converting MLX adapter to PEFT format...")

    mlx_path = Path(mlx_adapter_path)
    peft_path = Path(peft_output_path)
    peft_path.mkdir(parents=True, exist_ok=True)

    # Load MLX adapter weights
    mlx_weights = load_file(mlx_path / "adapters.safetensors")

    # Map MLX keys to PEFT keys
    peft_weights = {}
    for key, value in mlx_weights.items():
        new_key = f"base_model.model.model.{key}"
        new_key = new_key.replace(".lora_a.", ".lora_A.")
        new_key = new_key.replace(".lora_b.", ".lora_B.")
        peft_weights[new_key] = value

    # Save PEFT weights
    save_file(peft_weights, peft_path / "adapter_model.safetensors")

    # Create adapter_config.json
    config = {
        "base_model_name_or_path": "microsoft/Phi-3-mini-4k-instruct",
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "peft_type": "LORA",
        "r": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "task_type": "CAUSAL_LM",
    }

    with open(peft_path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"PEFT adapter saved to {peft_path}")


def merge_lora(base_model: str, lora_path: str, output_path: str):
    """Merge LoRA adapter into base model."""
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merging...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("Merge complete!")


def convert_to_gguf(model_path: str, output_path: str, convert_script: str = None):
    """Convert HF model to GGUF format."""
    print(f"Converting to GGUF: {model_path} -> {output_path}")

    # Find convert script
    candidates = [
        convert_script,
        Path("scripts/convert_hf_to_gguf.py"),
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        ]

    script = None
    for c in candidates:
        if c and Path(c).exists():
            script = str(c)
            break

    if not script:
        print("❌ convert_hf_to_gguf.py not found")
        print("Download from llama.cpp or provide --convert-script")
        return False

    result = subprocess.run([
        "python", script, model_path,
        "--outfile", output_path,
        "--outtype", "f16"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    print("GGUF conversion complete!")
    return True


def quantize_gguf(input_path: str, output_path: str, quant_type: str = "q4_K_M"):
    """Quantize GGUF model."""
    print(f"Quantizing: {input_path} -> {output_path}")

    quantize_bin = shutil.which("llama-quantize")
    if not quantize_bin:
        candidates = [
            Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize",
            Path("/opt/homebrew/bin/llama-quantize"),
            ]
        for c in candidates:
            if c.exists():
                quantize_bin = str(c)
                break

    if not quantize_bin:
        print("❌ llama-quantize not found")
        print(f"Run manually: llama-quantize {input_path} {output_path} {quant_type}")
        return False

    result = subprocess.run([quantize_bin, input_path, output_path, quant_type])

    if result.returncode != 0:
        return False

    print("Quantization complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export design-analyst to GGUF")
    parser.add_argument("--adapter", "-a", default="adapters/design-analyst-v4",
                        help="MLX adapter path")
    parser.add_argument("--output", "-o", default="models/design-analyst-v4",
                        help="Output base name")
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--convert-script", help="Path to convert_hf_to_gguf.py")
    parser.add_argument("--skip-convert-adapter", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)

    peft_path = f"{args.adapter}-peft"
    merged_path = f"{args.output}-merged"
    f16_path = f"{args.output}-f16.gguf"
    q4_path = f"{args.output}-q4.gguf"

    # Step 1: Convert MLX to PEFT
    if not args.skip_convert_adapter:
        convert_mlx_to_peft(args.adapter, peft_path)

    # Step 2: Merge LoRA
    if not args.skip_merge:
        merge_lora(args.base_model, peft_path, merged_path)

    # Step 3: Convert to GGUF
    if not args.skip_gguf:
        convert_to_gguf(merged_path, f16_path, args.convert_script)

    # Step 4: Quantize
    if not args.skip_quantize:
        quantize_gguf(f16_path, q4_path)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"Quantized model: {q4_path}")
    print(f"\nTest: llama-cli -m {q4_path} -p '<|user|>Hello<|end|><|assistant|>' -n 50")


if __name__ == "__main__":
    main()