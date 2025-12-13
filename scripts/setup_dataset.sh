#!/bin/bash
# Cyan Whiteboard Dataset Setup Script
# Run this on your M3 Max

set -e

# === CONFIGURE THESE ===
HEIC_SOURCE_DIR="$HOME/Downloads/Cyan"  # <-- Change this
DATASET_DIR="$HOME/cyan-yolo-dataset"
LABELS_ZIP="$HOME/Downloads/non_image_ds.zip"     # <-- Change if different

# === SETUP ===
echo "🔧 Setting up dataset at $DATASET_DIR"

mkdir -p "$DATASET_DIR"/images/train
mkdir -p "$DATASET_DIR"/images/val
mkdir -p "$DATASET_DIR"/labels/train
mkdir -p "$DATASET_DIR"/labels/val

# === STEP 1: Convert HEIC to JPG ===
echo "📸 Converting HEIC to JPG..."

# Check if sips is available (built into macOS)
if ! command -v sips &> /dev/null; then
    echo "❌ sips not found. Are you on macOS?"
    exit 1
fi

# Create temp dir for converted images
CONVERTED_DIR="$DATASET_DIR/converted_images"
mkdir -p "$CONVERTED_DIR"

# Convert all HEIC files
count=0
for heic in "$HEIC_SOURCE_DIR"/IMG_*.HEIC "$HEIC_SOURCE_DIR"/IMG_*.heic; do
    if [ -f "$heic" ]; then
        fname=$(basename "$heic" | sed 's/\.[hH][eE][iI][cC]$/.jpg/')
        sips -s format jpeg "$heic" --out "$CONVERTED_DIR/$fname" > /dev/null 2>&1
        count=$((count + 1))
    fi
done
echo "✅ Converted $count HEIC files to JPG"

# === STEP 2: Extract labels ===
echo "📦 Extracting labels..."

LABELS_TEMP="$DATASET_DIR/labels_temp"
mkdir -p "$LABELS_TEMP"
unzip -q -o "$LABELS_ZIP" -d "$LABELS_TEMP"

# === STEP 3: Match images with labels and split ===
echo "🔀 Matching images to labels and splitting train/val (85/15)..."

# Get list of annotated labels (non-empty .txt files)
annotated=()
for txt in "$LABELS_TEMP"/obj_Train_data/*.txt; do
    if [ -s "$txt" ]; then
        base=$(basename "$txt" .txt)
        img="$CONVERTED_DIR/${base}.jpg"
        if [ -f "$img" ]; then
            annotated+=("$base")
        fi
    fi
done

echo "Found ${#annotated[@]} images with annotations"

# Shuffle and split
shuf_annotated=($(printf '%s\n' "${annotated[@]}" | shuf))
total=${#shuf_annotated[@]}
val_count=$((total * 15 / 100))
train_count=$((total - val_count))

echo "Train: $train_count, Val: $val_count"

# Copy train
i=0
while [ $i -lt $train_count ]; do
    base="${shuf_annotated[$i]}"
    cp "$CONVERTED_DIR/${base}.jpg" "$DATASET_DIR/images/train/"
    cp "$LABELS_TEMP/obj_Train_data/${base}.txt" "$DATASET_DIR/labels/train/"
    i=$((i + 1))
done

# Copy val
while [ $i -lt $total ]; do
    base="${shuf_annotated[$i]}"
    cp "$CONVERTED_DIR/${base}.jpg" "$DATASET_DIR/images/val/"
    cp "$LABELS_TEMP/obj_Train_data/${base}.txt" "$DATASET_DIR/labels/val/"
    i=$((i + 1))
done

# === STEP 4: Create dataset.yaml ===
echo "📝 Creating dataset.yaml..."

cat > "$DATASET_DIR/dataset.yaml" << EOF
# Cyan Whiteboard Shape Detection Dataset
path: $DATASET_DIR
train: images/train
val: images/val

names:
  0: rectangle
  1: rounded_rectangle
  2: oval
  3: circle
  4: diamond
  5: triangle
  6: cylinder
  7: cloud
  8: hexagon
  9: parallelogram
  10: sticky_note
  11: stick_figure
  12: solid_arrow
  13: dashed_arrow
  14: bidirectional_arrow
  15: line
  16: curved_arrow
  17: start_dot
  18: end_dot
  19: text_label
  20: ellipse
  21: square
  22: curved_bidirectional_arrow
  23: dashed_line
  24: dotted_line
  25: dotted_arrow
  26: solid_circle
  27: double_solid_line
  28: dashed_oval
  29: curved_line
EOF

# === STEP 5: Cleanup ===
echo "🧹 Cleaning up..."
rm -rf "$LABELS_TEMP"

# === SUMMARY ===
echo ""
echo "✅ Dataset ready at: $DATASET_DIR"
echo ""
echo "Structure:"
find "$DATASET_DIR" -type d | head -20
echo ""
echo "Counts:"
echo "  Train images: $(ls "$DATASET_DIR/images/train" | wc -l)"
echo "  Train labels: $(ls "$DATASET_DIR/labels/train" | wc -l)"
echo "  Val images:   $(ls "$DATASET_DIR/images/val" | wc -l)"
echo "  Val labels:   $(ls "$DATASET_DIR/labels/val" | wc -l)"
echo ""
echo "📋 Next: Run training with:"
echo "  pip install ultralytics"
echo "  yolo detect train model=yolov8n.pt data=$DATASET_DIR/dataset.yaml epochs=100 imgsz=640"