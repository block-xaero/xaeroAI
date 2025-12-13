---
name: whiteboard-detector
version: 0.1.0
kind: onnx
tags:
  - vision
  - detection
  - whiteboard
  - shapes
  - cyan-core
capabilities:
  - image_to_boxes
input:
  type: image
  formats:
    - png
    - jpeg
output:
  type: boxes
  schema:
    type: array
    items:
      type: object
      properties:
        class_id: { type: integer }
        class_name: { type: string }
        confidence: { type: number }
        bbox: { type: array, items: { type: number }, minItems: 4, maxItems: 4 }
base_model: null
lora_rank: null
author: cyan
created: 1734048000
model_file: whiteboard-yolov8n.onnx
---

# Whiteboard Shape Detector

Detects 30 shape classes from whiteboard photos using YOLOv8-nano.

## Overview

This model is trained on hand-drawn whiteboard images to detect common diagramming shapes including flowchart elements, UML components, and connectors.

## Classes (30)

1. rectangle
2. rounded_rectangle
3. oval
4. circle
5. diamond
6. triangle
7. cylinder
8. cloud
9. hexagon
10. parallelogram
11. sticky_note
12. stick_figure
13. solid_arrow
14. dashed_arrow
15. bidirectional_arrow
16. line
17. curved_arrow
18. start_dot
19. end_dot
20. text_label
21. ellipse
22. square
23. curved_bidirectional_arrow
24. dashed_line
25. dotted_line
26. dotted_arrow
27. solid_circle
28. double_solid_line
29. dashed_oval
30. curved_line

## Usage

```python
# Input: JPEG or PNG image bytes
# Output: List of detected boxes with class, confidence, and bbox

boxes = model.detect(image_bytes)
for box in boxes:
    print(f"{box.class_name}: {box.confidence:.2f} at {box.bbox}")
```

## Performance

- Input size: 640x640
- Inference: ~2ms on M3 Max (Metal)
- mAP@0.5: TBD (training in progress)

## Training Data

- 211 annotated whiteboard images
- Hand-drawn diagrams with various styles
- Multiple shape combinations per image
