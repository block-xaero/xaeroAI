#!/usr/bin/env python3
"""
Download all models needed for the xaeroai pipeline.

Usage:
    python download_models.py --models-dir ./models

Downloads:
    1. TrOCR (ONNX) - OCR model
    2. cyan-lens (GGUF) - Phi-3 + LoRA for Mermaid generation

You still need to copy your YOLO model:
    cp /path/to/runs/detect/train/weights/best.onnx ./models/whiteboard-detector/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def download_ocr(models_dir: Path):
    """
    Download PaddleOCR ONNX models from monkt/paddleocr-onnx.
    
    PaddleOCR is production-ready:
    - 60k+ GitHub stars
    - Used by MinerU, RAGFlow, major enterprises
    - Apache 2.0 license
    - CTC-based = single forward pass (no decoder loop)
    - 85%+ accuracy on English
    """
    print("\n" + "="*60)
    print("📥 Downloading PaddleOCR (ONNX)...")
    print("="*60)
    
    ocr_dir = models_dir / "paddleocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    rec_path = ocr_dir / "rec.onnx"
    if rec_path.exists():
        print(f"✅ PaddleOCR already exists at {ocr_dir}")
        return
    
    repo_id = "monkt/paddleocr-onnx"
    
    print(f"Downloading from {repo_id}...")
    print("   - Detection model (PP-OCRv5): 84 MB")
    print("   - Recognition model (English): 7.5 MB")
    print()
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Download detection model
        det_path = hf_hub_download(
            repo_id=repo_id,
            filename="detection/v5/det.onnx",
            local_dir=str(ocr_dir),
            local_dir_use_symlinks=False
        )
        print(f"   ✓ Detection model")
        
        # Download recognition model (English)
        rec_path = hf_hub_download(
            repo_id=repo_id,
            filename="languages/english/rec.onnx",
            local_dir=str(ocr_dir),
            local_dir_use_symlinks=False
        )
        print(f"   ✓ Recognition model")
        
        # Download dictionary
        dict_path = hf_hub_download(
            repo_id=repo_id,
            filename="languages/english/dict.txt",
            local_dir=str(ocr_dir),
            local_dir_use_symlinks=False
        )
        print(f"   ✓ Dictionary")
        
        # Flatten structure for easier access
        import shutil
        final_det = ocr_dir / "det.onnx"
        final_rec = ocr_dir / "rec.onnx"
        final_dict = ocr_dir / "dict.txt"
        
        if not final_det.exists():
            shutil.copy(ocr_dir / "detection" / "v5" / "det.onnx", final_det)
        if not final_rec.exists():
            shutil.copy(ocr_dir / "languages" / "english" / "rec.onnx", final_rec)
        if not final_dict.exists():
            shutil.copy(ocr_dir / "languages" / "english" / "dict.txt", final_dict)
        
        # Clean up nested dirs
        shutil.rmtree(ocr_dir / "detection", ignore_errors=True)
        shutil.rmtree(ocr_dir / "languages", ignore_errors=True)
        
        print(f"✅ PaddleOCR saved to {ocr_dir}")
        
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)
    
    # Create SKILL.md
    skill_md = ocr_dir / "SKILL.md"
    skill_md.write_text("""---
name: paddleocr
version: 5.0.0
kind: onnx
tags:
  - ocr
  - ctc
  - paddleocr
  - production
capabilities:
  - image_to_text
input:
  type: image
  formats:
    - png
    - jpeg
output:
  type: text
base_model: PaddlePaddle/PP-OCRv5
author: PaddlePaddle
det_model_file: det.onnx
rec_model_file: rec.onnx
dict_file: dict.txt
---

# PaddleOCR (PP-OCRv5)

Production-ready OCR from PaddlePaddle. 60k+ GitHub stars.
Used by MinerU, RAGFlow, and major enterprises.

## Architecture

CTC-based recognition (single forward pass):

```
Image → Detection → Crop regions → Recognition → CTC Decode → Text
```

Unlike TrOCR (encoder-decoder), PaddleOCR recognition is:
- Single ONNX forward pass
- CTC greedy/beam decode (pure algorithm, no neural network loop)
- Fast: ~10-50ms per text region

## Models

| File | Size | Purpose |
|------|------|---------|
| det.onnx | 84 MB | Text detection (finds text regions) |
| rec.onnx | 7.5 MB | Text recognition (reads text) |
| dict.txt | ~10 KB | Character vocabulary |

## Inference Flow

```python
# 1. Detection: find text boxes
det_input = preprocess_for_det(image)  # resize, normalize
det_output = det_session.run(det_input)  # [H, W] probability map
boxes = postprocess_det(det_output)  # threshold + contour finding

# 2. Recognition: read each box
for box in boxes:
    cropped = crop_and_resize(image, box, height=32)
    rec_input = preprocess_for_rec(cropped)  # normalize
    rec_output = rec_session.run(rec_input)  # [T, vocab_size] logits
    text = ctc_decode(rec_output, dict)  # greedy or beam search
```

## Accuracy

| Language | Accuracy | Dataset |
|----------|----------|---------|
| English | 85.25% | 6,530 images |
| Latin (32 langs) | 84.7% | 3,111 images |
| Korean | 88.0% | 5,007 images |
| Greek | 89.28% | 2,799 images |

## For Whiteboard OCR

Since we already detect shapes with YOLO, we skip the detection model
and only use recognition on cropped shape regions:

```
YOLO box → crop image → rec.onnx → CTC decode → text
```

## License

Apache 2.0 - Commercial use allowed.
""")
    print(f"✅ Created {skill_md}")


def download_cyan_lens(models_dir: Path):
    """Download cyan-lens GGUF from HuggingFace."""
    print("\n" + "="*60)
    print("📥 Downloading cyan-lens (Phi-3 + LoRA)...")
    print("="*60)
    
    cyan_dir = models_dir / "cyan-lens"
    cyan_dir.mkdir(parents=True, exist_ok=True)
    
    gguf_path = cyan_dir / "cyan-lens-q4.gguf"
    if gguf_path.exists():
        print(f"✅ cyan-lens already exists at {gguf_path}")
        return
    
    # Use huggingface-cli to download
    print("Downloading from blockxaero/cyan-lens...")
    
    try:
        # Try huggingface_hub first
        from huggingface_hub import hf_hub_download
        
        downloaded = hf_hub_download(
            repo_id="blockxaero/cyan-lens",
            filename="cyan-lens-q4.gguf",
            local_dir=str(cyan_dir),
            local_dir_use_symlinks=False
        )
        print(f"✅ Downloaded to {downloaded}")
        
    except ImportError:
        # Fall back to CLI
        print("Using huggingface-cli...")
        result = subprocess.run([
            "huggingface-cli", "download",
            "blockxaero/cyan-lens",
            "cyan-lens-q4.gguf",
            "--local-dir", str(cyan_dir)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Download failed: {result.stderr}")
            print("\nManual download:")
            print("  1. Go to https://huggingface.co/blockxaero/cyan-lens")
            print("  2. Download cyan-lens-q4.gguf")
            print(f"  3. Place in {cyan_dir}/")
            sys.exit(1)
        
        print(f"✅ Downloaded to {cyan_dir}")
    
    # Create SKILL.md
    skill_md = cyan_dir / "SKILL.md"
    skill_md.write_text("""---
name: cyan-lens
version: 0.1.0
kind: gguf
tags:
  - llm
  - mermaid
  - phi-3
  - lora
capabilities:
  - text_generation
  - text_to_mermaid
input:
  type: text
  formats: []
output:
  type: text
base_model: microsoft/Phi-3-mini-4k-instruct
lora_rank: 8
author: blockxaero
model_file: cyan-lens-q4.gguf
---

# Cyan Lens

Phi-3 + LoRA for Mermaid diagram generation.

## Prompt Format

```
<|user|>
Your prompt here
<|end|>
<|assistant|>
```
""")
    print(f"✅ Created {skill_md}")


def setup_yolo_placeholder(models_dir: Path):
    """Create placeholder for YOLO model."""
    print("\n" + "="*60)
    print("📋 YOLO Setup (manual)")
    print("="*60)
    
    yolo_dir = models_dir / "whiteboard-detector"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = yolo_dir / "best.onnx"
    if onnx_path.exists():
        print(f"✅ YOLO model exists at {onnx_path}")
        return
    
    print(f"""
⚠️  YOLO model not found. Please copy manually:

    cp /Users/anirudhvyas/xaeroai/scripts/runs/detect/train/weights/best.onnx \\
       {yolo_dir}/best.onnx
""")
    
    # Create SKILL.md anyway
    skill_md = yolo_dir / "SKILL.md"
    skill_md.write_text("""---
name: whiteboard-detector
version: 0.1.0
kind: onnx
tags:
  - yolo
  - detection
  - whiteboard
capabilities:
  - image_to_boxes
input:
  type: image
  formats:
    - png
    - jpeg
output:
  type: boxes
base_model: yolov8n
author: blockxaero
model_file: best.onnx
---

# Whiteboard Detector

YOLOv8-nano for whiteboard shape detection.

## Classes (30)

rectangle, rounded_rectangle, oval, circle, diamond, hexagon,
parallelogram, triangle, star, cloud, cylinder, stick_figure,
arrow_box, document_shape, database_icon, square, ellipse,
pentagon, cross, heart, lightning, banner, callout, bracket,
solid_arrow, dashed_arrow, bidirectional_arrow, dotted_line,
curved_arrow, curved_line
""")
    
    # Create classes.txt
    classes_txt = yolo_dir / "classes.txt"
    classes_txt.write_text("""rectangle
rounded_rectangle
oval
circle
diamond
hexagon
parallelogram
triangle
star
cloud
cylinder
stick_figure
arrow_box
document_shape
database_icon
square
ellipse
pentagon
cross
heart
lightning
banner
callout
bracket
solid_arrow
dashed_arrow
bidirectional_arrow
dotted_line
curved_arrow
curved_line
""")
    print(f"✅ Created {skill_md}")
    print(f"✅ Created {classes_txt}")


def main():
    parser = argparse.ArgumentParser(description="Download xaeroai models")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("./models"),
        help="Directory to store models (default: ./models)"
    )
    parser.add_argument(
        "--skip-trocr",
        action="store_true",
        help="Skip OCR download"
    )
    parser.add_argument(
        "--skip-cyan",
        action="store_true",
        help="Skip cyan-lens download"
    )
    
    args = parser.parse_args()
    
    print("🚀 xaeroai Model Downloader")
    print(f"   Target: {args.models_dir.absolute()}")
    
    args.models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download each model
    if not args.skip_trocr:
        download_ocr(args.models_dir)
    
    if not args.skip_cyan:
        download_cyan_lens(args.models_dir)
    
    setup_yolo_placeholder(args.models_dir)
    
    print("\n" + "="*60)
    print("✅ Done!")
    print("="*60)
    print(f"""
Models directory structure:
{args.models_dir}/
├── whiteboard-detector/
│   ├── best.onnx         ← Copy your YOLO model here
│   ├── classes.txt
│   └── SKILL.md
├── paddleocr/
│   ├── det.onnx          # 84 MB - text detection (not used - YOLO does detection)
│   ├── rec.onnx          # 7.5 MB - text recognition
│   ├── dict.txt          # character vocabulary
│   └── SKILL.md
└── cyan-lens/
    ├── cyan-lens-q4.gguf
    └── SKILL.md
""")


if __name__ == "__main__":
    main()
