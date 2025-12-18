#!/bin/bash
# download_models.sh
# Downloads AI models from HuggingFace for Cyan + xaeroai
#
# Directory structure required by xaeroai:
#   models_dir/
#   ├── cyan-sketch/         # YOLO detector (blockxaero/cyan-sketch)
#   │   ├── SKILL.md
#   │   ├── best.onnx        # <- renamed from model.onnx
#   │   └── classes.txt
#   ├── paddleocr/           # OCR recognition
#   │   ├── SKILL.md
#   │   ├── inference.onnx   # <- converted from PaddlePaddle format
#   │   └── dict.txt
#   └── cyan-lens/           # Phi-3 fine-tuned (blockxaero/cyan-lens)
#       ├── SKILL.md
#       └── phi-3-mini-Q4.gguf

set -e

MODELS_DIR="${1:-$HOME/Documents/CyanModels}"
CONTAINER_DIR="$HOME/Library/Containers/6E1945B9-5A6A-49BF-9826-E1F4A7D5AF89/Data/Documents/CyanModels"

echo "📦 CyanLens Model Setup"
echo "   Source: $MODELS_DIR"
echo "   Target: $CONTAINER_DIR"
echo ""

mkdir -p "$MODELS_DIR"

# =============================================================================
# Helper Functions
# =============================================================================

check_and_download_hf() {
    local repo="$1"
    local dest="$2"
    local marker_file="$3"

    if [ -f "$dest/$marker_file" ]; then
        echo "   ✓ Already exists: $marker_file"
        return 0
    fi

    echo "   Downloading from HuggingFace: $repo..."
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$repo" \
            --local-dir "$dest" \
            --local-dir-use-symlinks False \
            2>/dev/null || echo "   ⚠️ HuggingFace download failed"
    else
        echo "   ⚠️ huggingface-cli not found. Install with: pip install huggingface_hub"
    fi
}

# =============================================================================
# 1. YOLO Whiteboard Detector (cyan-sketch)
# =============================================================================
echo "🔍 [1/3] cyan-sketch (YOLO whiteboard detector)..."

mkdir -p "$MODELS_DIR/cyan-sketch"

# Download if needed
check_and_download_hf "blockxaero/cyan-sketch" "$MODELS_DIR/cyan-sketch" "model.onnx"

# Rename model.onnx -> best.onnx if needed
if [ -f "$MODELS_DIR/cyan-sketch/model.onnx" ] && [ ! -f "$MODELS_DIR/cyan-sketch/best.onnx" ]; then
    echo "   Renaming model.onnx -> best.onnx"
    mv "$MODELS_DIR/cyan-sketch/model.onnx" "$MODELS_DIR/cyan-sketch/best.onnx"
elif [ -f "$MODELS_DIR/cyan-sketch/best.onnx" ]; then
    echo "   ✓ best.onnx exists"
fi

# Create SKILL.md if missing
if [ ! -f "$MODELS_DIR/cyan-sketch/SKILL.md" ]; then
    echo "   Creating SKILL.md..."
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
model_file: best.onnx
---

# Cyan Sketch - Whiteboard Shape Detector

YOLOv8-based detector trained on whiteboard photos.
Detects 30 shape classes including rectangles, diamonds, arrows, etc.
EOF
fi

# =============================================================================
# 2. PaddleOCR Recognition
# =============================================================================
echo ""
echo "🔍 [2/3] paddleocr (OCR recognition)..."

mkdir -p "$MODELS_DIR/paddleocr"

# Check if we already have the ONNX version
if [ -f "$MODELS_DIR/paddleocr/inference.onnx" ]; then
    echo "   ✓ inference.onnx exists"
else
    echo "   Downloading PaddleOCR ONNX model from monkt/paddleocr-onnx..."

    # Download from monkt/paddleocr-onnx (reliable, pre-converted)
    if curl -fSL "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/english/rec.onnx" -o "$MODELS_DIR/paddleocr/inference.onnx" 2>/dev/null; then
        # Verify file size (should be > 1MB)
        FILE_SIZE=$(stat -f%z "$MODELS_DIR/paddleocr/inference.onnx" 2>/dev/null || stat -c%s "$MODELS_DIR/paddleocr/inference.onnx" 2>/dev/null)
        if [ "$FILE_SIZE" -gt 1000000 ]; then
            echo "   ✓ Downloaded inference.onnx ($(( FILE_SIZE / 1024 / 1024 ))MB)"
        else
            echo "   ❌ Download failed (file too small)"
            rm -f "$MODELS_DIR/paddleocr/inference.onnx"
        fi
    else
        echo "   ❌ Failed to download from HuggingFace"
    fi
fi

# Download character dictionary
if [ -f "$MODELS_DIR/paddleocr/dict.txt" ]; then
    echo "   ✓ dict.txt exists"
else
    echo "   Downloading OCR dictionary..."
    if curl -fSL "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/english/dict.txt" -o "$MODELS_DIR/paddleocr/dict.txt" 2>/dev/null; then
        echo "   ✓ Downloaded dict.txt"
    else
        # Fallback to PaddleOCR GitHub
        curl -fSL "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/en_dict.txt" \
            -o "$MODELS_DIR/paddleocr/dict.txt" 2>/dev/null || echo "   ⚠️ Failed to download dict.txt"
    fi
fi

# Create SKILL.md if missing
if [ ! -f "$MODELS_DIR/paddleocr/SKILL.md" ]; then
    echo "   Creating SKILL.md..."
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
fi

# =============================================================================
# 3. Cyan Lens (Phi-3 fine-tuned)
# =============================================================================
echo ""
echo "🔍 [3/3] cyan-lens (Phi-3 fine-tuned for diagrams)..."

mkdir -p "$MODELS_DIR/cyan-lens"

# Check for existing GGUF file
if [ -f "$MODELS_DIR/cyan-lens/phi-3-mini-Q4.gguf" ]; then
    echo "   ✓ phi-3-mini-Q4.gguf exists"
else
    check_and_download_hf "blockxaero/cyan-lens" "$MODELS_DIR/cyan-lens" "phi-3-mini-Q4.gguf"
fi

# Create SKILL.md if missing
if [ ! -f "$MODELS_DIR/cyan-lens/SKILL.md" ]; then
    echo "   Creating SKILL.md..."
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
# Copy to Container
# =============================================================================
echo ""
echo "📋 Copying to app container..."

mkdir -p "$CONTAINER_DIR"

# Use rsync for efficient copying (only copies changed files)
if command -v rsync &> /dev/null; then
    rsync -av --progress "$MODELS_DIR/" "$CONTAINER_DIR/"
else
    cp -R "$MODELS_DIR"/* "$CONTAINER_DIR/"
fi

echo ""
echo "✅ Setup complete!"
echo ""

# =============================================================================
# Verification
# =============================================================================
echo "📁 Verifying required files..."

MISSING=""
[ ! -f "$CONTAINER_DIR/cyan-sketch/best.onnx" ] && MISSING="$MISSING cyan-sketch/best.onnx"
[ ! -f "$CONTAINER_DIR/cyan-sketch/SKILL.md" ] && MISSING="$MISSING cyan-sketch/SKILL.md"
[ ! -f "$CONTAINER_DIR/paddleocr/inference.onnx" ] && MISSING="$MISSING paddleocr/inference.onnx"
[ ! -f "$CONTAINER_DIR/paddleocr/SKILL.md" ] && MISSING="$MISSING paddleocr/SKILL.md"
[ ! -f "$CONTAINER_DIR/paddleocr/dict.txt" ] && MISSING="$MISSING paddleocr/dict.txt"
[ ! -f "$CONTAINER_DIR/cyan-lens/phi-3-mini-Q4.gguf" ] && MISSING="$MISSING cyan-lens/phi-3-mini-Q4.gguf"
[ ! -f "$CONTAINER_DIR/cyan-lens/SKILL.md" ] && MISSING="$MISSING cyan-lens/SKILL.md"

if [ -n "$MISSING" ]; then
    echo "❌ Missing files:$MISSING"
    echo ""
    exit 1
else
    echo "✓ All required files present"
fi

echo ""
echo "💾 Total size:"
du -sh "$CONTAINER_DIR"

echo ""
echo "📍 Container models: $CONTAINER_DIR"
echo ""
echo "Ready for Cyan! 🚀"