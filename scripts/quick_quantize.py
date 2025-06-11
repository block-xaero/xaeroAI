# scripts/fixed_quantize.py
"""
Fixed quantization script that handles different model types
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForVision2Seq,
    VisionEncoderDecoderModel,
    AutoConfig
)
from optimum.quanto import quantize, freeze, qint4, qint8
import safetensors.torch as safetensors
from pathlib import Path
import argparse
import json

def load_model_with_correct_class(model_path: str):
    """Load model with the correct AutoModel class"""
    
    # Check the config to determine model type
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config_class = config.__class__.__name__
    
    print(f"Model config type: {config_class}")
    
    if "VisionEncoderDecoder" in config_class:
        print("Loading as VisionEncoderDecoderModel...")
        model = VisionEncoderDecoderModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    elif any(causal_type in config_class for causal_type in [
        "GPT", "Llama", "CodeGen", "Phi", "Qwen", "Mistral", "Gemma"
    ]):
        print("Loading as AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    else:
        print(f"Trying AutoModelForCausalLM for {config_class}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    return model

def quantize_model_fixed(model_path: str, output_path: str, bits: int = 4):
    """Fixed quantization that handles different model types"""
    
    print(f"📥 Loading model from {model_path}")
    
    try:
        # Load model with correct class
        model = load_model_with_correct_class(model_path)
        
        # Get original size
        original_params = sum(p.numel() for p in model.parameters())
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        print(f"Original: {original_params:,} parameters, {original_size:.1f}MB")
        
        # Quantize with optimum.quanto
        print(f"🔄 Quantizing to {bits}-bit...")
        if bits == 4:
            quantize(model, weights=qint4)
        elif bits == 8:
            quantize(model, weights=qint8)
        else:
            raise ValueError("Only 4-bit and 8-bit quantization supported")
        
        freeze(model)
        
        # Calculate quantized size
        quantized_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        compression_ratio = original_size / quantized_size
        
        print(f"Quantized: {quantized_size:.1f}MB ({compression_ratio:.1f}x compression)")
        
        # Save
        print(f"💾 Saving to {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as safetensors
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        safetensors.save_file(state_dict, output_path)
        
        # Save metadata
        metadata = {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": compression_ratio,
            "quantization_bits": bits,
            "parameter_count": original_params,
            "quantization_method": "optimum.quanto",
            "model_type": type(model).__name__
        }
        
        metadata_path = output_path.replace('.safetensors', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Quantization complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error quantizing {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def quantize_all_foundation_models():
    """Quantize all downloaded foundation models"""
    
    foundation_dir = Path("models/foundation")
    quantized_dir = Path("models/quantized")
    quantized_dir.mkdir(parents=True, exist_ok=True)
    
    if not foundation_dir.exists():
        print(f"❌ Foundation models directory not found: {foundation_dir}")
        return
    
    # Find all model directories
    model_dirs = []
    for category_dir in foundation_dir.iterdir():
        if category_dir.is_dir():
            for model_dir in category_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "config.json").exists():
                    model_dirs.append(model_dir)
    
    if not model_dirs:
        print(f"❌ No models found in {foundation_dir}")
        return
    
    print(f"🔍 Found {len(model_dirs)} models to quantize:")
    for model_dir in model_dirs:
        print(f"   - {model_dir.name}")
    print()
    
    successful = 0
    failed = 0
    
    for i, model_dir in enumerate(model_dirs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(model_dirs)}] Quantizing {model_dir.name}")
        print(f"{'='*60}")
        
        # Quantize to 4-bit
        output_4bit = quantized_dir / f"{model_dir.name}_q4.safetensors"
        if quantize_model_fixed(str(model_dir), str(output_4bit), bits=4):
            successful += 1
        else:
            failed += 1
    
    print(f"\n🎉 Quantization Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 Output directory: {quantized_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed model quantization")
    parser.add_argument("--all", action="store_true", help="Quantize all foundation models")
    parser.add_argument("--model", help="Path to specific model")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    
    args = parser.parse_args()
    
    if args.all:
        quantize_all_foundation_models()
    elif args.model and args.output:
        quantize_model_fixed(args.model, args.output, args.bits)
    else:
        print("Usage: python scripts/fixed_quantize.py --all")
