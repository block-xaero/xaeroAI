---
license: other
license_name: business-source-license
license_link: LICENSE
tags:
  - yolo
  - object-detection
  - whiteboard
  - diagram
  - flowchart
  - onnx
library_name: onnxruntime
---

# Cyan Sketch - Whiteboard Shape Detector

YOLOv8n model for detecting shapes and connectors in whiteboard/flowchart images.

## Model Details

- **Architecture**: YOLOv8n (nano)
- **Format**: ONNX
- **Input Size**: 640x640
- **Classes**: 30 shape types

## Performance

| Metric | Value |
|--------|-------|
| mAP50 | 0.592 |
| mAP50-95 | 0.339 |

### Per-Class Performance (Top 10)

| Class | mAP50 |
|-------|-------|
| rounded_rectangle | 0.995 |
| stick_figure | 0.995 |
| cloud | 0.980 |
| rectangle | 0.857 |
| sticky_note | 0.857 |
| cylinder | 0.823 |
| text_label | 0.774 |
| circle | 0.738 |
| oval | 0.735 |
| diamond | 0.713 |

## Classes (30)
```
rectangle, rounded_rectangle, oval, circle, diamond, triangle,
cylinder, cloud, hexagon, parallelogram, sticky_note, stick_figure,
solid_arrow, dashed_arrow, bidirectional_arrow, line, curved_arrow,
start_dot, end_dot, text_label, ellipse, square,
curved_bidirectional_arrow, dashed_line, dotted_line, dotted_arrow,
solid_circle, double_solid_line, dashed_oval, curved_line
```

## Usage
```python
import onnxruntime as ort
import cv2
import numpy as np

# Load model
session = ort.InferenceSession("best.onnx")

# Load classes
with open("classes.txt") as f:
    classes = [l.strip() for l in f]

# Preprocess image
img = cv2.imread("whiteboard.jpg")
resized = cv2.resize(img, (640, 640))
blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
blob = np.transpose(blob, (2, 0, 1))[None, ...]

# Run inference
outputs = session.run(None, {"images": blob})[0]

# Parse detections (conf > 0.3)
for i in range(outputs.shape[2]):
    scores = outputs[0, 4:, i]
    class_id = np.argmax(scores)
    conf = scores[class_id]
    if conf > 0.3:
        print(f"{classes[class_id]}: {conf:.2f}")
```

## Files

- `best.onnx` - ONNX model (6MB)
- `classes.txt` - Class names
- `ocr_dictionary.json` - Domain terms for OCR correction

## License

Business Source License - See LICENSE file
