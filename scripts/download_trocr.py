#!/usr/bin/env python3
"""
Download TrOCR and convert to ONNX for xaeroai.

Usage:
    pip install transformers optimum onnx onnxruntime
    python download_trocr.py

Output:
    models/trocr-small/
    ├── model.onnx
    ├── SKILL.md
    └── config.json
"""

import os
from pathlib import Path

def main():
    output_dir = Path("models/trocr-small")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📦 Exporting TrOCR to ONNX...")
    
    # Use optimum for direct ONNX export
    from optimum.onnxruntime import ORTModelForVision2Seq
    from transformers import TrOCRProcessor
    
    model_id = "microsoft/trocr-small-handwritten"
    
    # Export to ONNX
    model = ORTModelForVision2Seq.from_pretrained(
        model_id,
        export=True
    )
    model.save_pretrained(output_dir)
    
    # Save processor
    processor = TrOCRProcessor.from_pretrained(model_id)
    processor.save_pretrained(output_dir)
    
    print(f"✅ Model saved to {output_dir}")
    
    # Create SKILL.md
    skill_md = '''---
name: trocr-small
version: 1.0.0
kind: onnx
tags:
  - vision
  - ocr
  - handwriting
  - text-recognition
capabilities:
  - image_to_text
input:
  type: image
  formats:
    - png
    - jpeg
output:
  type: text
  formats: []
base_model: microsoft/trocr-small-handwritten
author: microsoft
created: 1734048000
model_file: model.onnx
---

# TrOCR Small Handwritten

Microsoft's TrOCR model fine-tuned for handwritten text recognition.

## Overview

Transformer-based OCR model that combines a ViT image encoder with a 
text decoder. Pretrained on handwritten text datasets.

## Usage

```python
# Input: Cropped image of text region (PNG/JPEG bytes)
# Output: Recognized text string

text = model.recognize(image_bytes)
```

## Performance

- Encoder: ViT-small (384 hidden)
- Decoder: 6-layer transformer
- Inference: ~50ms on M3 Max (ONNX + Metal)

## Notes

- Works best on single-line text crops
- Combine with dictionary post-processing for better accuracy
- Consider fine-tuning LoRA for personal handwriting style
'''
    
    (output_dir / "SKILL.md").write_text(skill_md)
    print(f"✅ SKILL.md created")
    
    # List output
    print(f"\n📁 Contents of {output_dir}:")
    for f in output_dir.iterdir():
        size = f.stat().st_size / 1024 / 1024
        print(f"   {f.name}: {size:.1f} MB")

if __name__ == "__main__":
    main()
