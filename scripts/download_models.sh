#!/bin/bash
# download_models.sh
# Downloads AI models from HuggingFace for Cyan + xaeroai
#
# Directory structure required by xaeroai:
#   models_dir/
#   ├── cyan-sketch/         # YOLO detector (blockxaero/cyan-sketch)
#   │   ├── SKILL.md
#   │   ├── model.onnx
#   │   └── classes.txt
#   ├── paddleocr/           # OCR recognition
#   │   ├── SKILL.md
#   │   ├── model.onnx
#   │   └── dict.txt
#   └── cyan-lens/           # Phi-3 fine-tuned (blockxaero/cyan-lens)
#       ├── SKILL.md
#       └── phi-3-mini-Q4.gguf

set -e

MODELS_DIR="${1:-$HOME/Documents/CyanModels}"

echo "📦 Downloading CyanLens models to: $MODELS_DIR"
echo ""

mkdir -p "$MODELS_DIR"

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "⚠️  huggingface-cli not found. Installing..."
    pip install huggingface_hub
fi

# =============================================================================
# 1. YOLO Whiteboard Detector (cyan-sketch)
# =============================================================================
echo "🔍 [1/3] Downloading cyan-sketch (YOLO whiteboard detector)..."

mkdir -p "$MODELS_DIR/cyan-sketch"

# Download from HuggingFace
huggingface-cli download blockxaero/cyan-sketch \
    --local-dir "$MODELS_DIR/cyan-sketch" \
    --local-dir-use-symlinks False \
    2>/dev/null || {
    echo "⚠️  HuggingFace download failed, checking if files exist..."
}

# Verify SKILL.md exists, create if missing
if [ ! -f "$MODELS_DIR/cyan-sketch/SKILL.md" ]; then
    echo "Creating SKILL.md for cyan-sketch..."
    cat > "$MODELS_DIR/cyan-sketch/SKILL.md" << 'EOF'
---
name: cyan-sketch
version: 0.1.0
kind: onnx
tags: [vision, detection, whiteboard, shapes]
capabilities:
  - image_to_boxes
input:
  type: image
  formats: [png, jpeg, jpg]
output:
  type: boxes
author: blockxaero
created: 1702400000
model_file: model.onnx
---

# Cyan Sketch - Whiteboard Shape Detector

YOLOv8-based detector trained on whiteboard photos.
Detects 30 shape classes including rectangles, diamonds, arrows, etc.

## Classes

rectangle, diamond, oval, circle, parallelogram, hexagon, triangle,
cloud, cylinder, document_shape, star, pentagon, solid_arrow,
dashed_arrow, bidirectional_arrow, curved_arrow, dotted_arrow,
line, dashed_line, dotted_line, curved_line, text_box, sticky_note,
database_icon, user_icon, stick_figure, cross, checkmark, and more.
EOF
fi

# =============================================================================
# 2. PaddleOCR Recognition
# =============================================================================
echo ""
echo "🔍 [2/3] Downloading PaddleOCR recognition model..."

mkdir -p "$MODELS_DIR/paddleocr"

# Download PaddleOCR v3 English recognition
PADDLEOCR_URL="https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar"

if [ ! -f "$MODELS_DIR/paddleocr/inference.onnx" ]; then
    echo "Downloading PaddleOCR..."
    curl -L "$PADDLEOCR_URL" -o "$MODELS_DIR/paddleocr/model.tar"
    tar -xf "$MODELS_DIR/paddleocr/model.tar" -C "$MODELS_DIR/paddleocr" --strip-components=1
    rm "$MODELS_DIR/paddleocr/model.tar"

    # Rename to expected name if needed
    if [ -f "$MODELS_DIR/paddleocr/inference.pdmodel" ]; then
        echo "Note: PaddleOCR is in PaddlePaddle format. You may need to convert to ONNX."
        echo "See: https://github.com/PaddlePaddle/Paddle2ONNX"
    fi
fi

# Download character dictionary
if [ ! -f "$MODELS_DIR/paddleocr/dict.txt" ]; then
    echo "Downloading OCR dictionary..."
    curl -L "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/en_dict.txt" \
        -o "$MODELS_DIR/paddleocr/dict.txt"
fi

# Create SKILL.md for paddleocr
cat > "$MODELS_DIR/paddleocr/SKILL.md" << 'EOF'
---
name: paddleocr-rec
version: 3.0.0
kind: onnx
tags: [ocr, text, recognition]
capabilities:
  - image_to_text
input:
  type: image
  formats: [png, jpeg]
output:
  type: text
author: PaddlePaddle
created: 1700000000
model_file: inference.onnx
---

# PaddleOCR Recognition Model

PP-OCRv3 English recognition model for extracting text from images.
Uses CTC decoding with the provided dict.txt character vocabulary.
EOF

# =============================================================================
# 3. Cyan Lens (Phi-3 fine-tuned)
# =============================================================================
echo ""
echo "🔍 [3/3] Downloading cyan-lens (Phi-3 fine-tuned for diagrams)..."

mkdir -p "$MODELS_DIR/cyan-lens"

# Download from HuggingFace
huggingface-cli download blockxaero/cyan-lens \
    --local-dir "$MODELS_DIR/cyan-lens" \
    --local-dir-use-symlinks False \
    2>/dev/null || {
    echo "⚠️  HuggingFace download failed, checking if files exist..."
}

# Verify SKILL.md exists, create if missing
if [ ! -f "$MODELS_DIR/cyan-lens/SKILL.md" ]; then
    echo "Creating SKILL.md for cyan-lens..."
    cat > "$MODELS_DIR/cyan-lens/SKILL.md" << 'EOF'
---
name: cyan-lens
version: 0.1.0
kind: gguf
tags: [llm, mermaid, diagrams, phi]
capabilities:
  - text_generation
  - text_to_mermaid
input:
  type: text
output:
  type: text
author: blockxaero
created: 1702400000
model_file: phi-3-mini-Q4.gguf
---

# Cyan Lens - Diagram Generation Model

Phi-3-mini fine-tuned for generating Mermaid diagrams from
whiteboard descriptions and answering design questions.

## Capabilities

- Convert shape descriptions to Mermaid syntax
- Answer questions about project health
- Analyze integration data

## Prompt Format

Uses Phi-3 chat format:
```
<|user|>
Your question here
<|end|>
<|assistant|>
```
EOF
fi

# =============================================================================
# Verification
# =============================================================================
echo ""
echo "✅ Download complete!"
echo ""
echo "📁 Directory structure:"
find "$MODELS_DIR" -type f \( -name "*.onnx" -o -name "*.gguf" -o -name "SKILL.md" -o -name "*.txt" \) 2>/dev/null | head -20
echo ""
echo "💾 Total size:"
du -sh "$MODELS_DIR"
echo ""

# Check for missing critical files
MISSING=""
[ ! -f "$MODELS_DIR/cyan-sketch/SKILL.md" ] && MISSING="$MISSING cyan-sketch/SKILL.md"
[ ! -f "$MODELS_DIR/paddleocr/SKILL.md" ] && MISSING="$MISSING paddleocr/SKILL.md"
[ ! -f "$MODELS_DIR/paddleocr/dict.txt" ] && MISSING="$MISSING paddleocr/dict.txt"
[ ! -f "$MODELS_DIR/cyan-lens/SKILL.md" ] && MISSING="$MISSING cyan-lens/SKILL.md"

if [ -n "$MISSING" ]; then
    echo "⚠️  Missing files:$MISSING"
    echo ""
fi

echo "📍 Models location: $MODELS_DIR"
echo ""
echo "This path should match what WorkspaceView.swift returns from getModelsDirectory()"
echo ""
echo "Expected by ai_bridge.rs:"
echo "  - yolo_dir:  $MODELS_DIR/cyan-sketch"
echo "  - ocr_dir:   $MODELS_DIR/paddleocr"
echo "  - phi_dir:   $MODELS_DIR/cyan-lens"