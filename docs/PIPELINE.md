# Pipeline: Whiteboard to Mermaid

## Overview

The pipeline converts a photo of a whiteboard into a valid Mermaid diagram.

```
📸 Phone Photo
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: DETECTION                              │
│                                                                         │
│  Model: whiteboard-detector (YOLO, ONNX)                               │
│  Input: Full image (640×640 resized)                                   │
│  Output: Bounding boxes with class labels                              │
│                                                                         │
│  Example output:                                                        │
│    Box 0: "rectangle" at (50, 30), size 80×40, conf 0.94               │
│    Box 1: "solid_arrow" at (130, 45), size 70×10, conf 0.87            │
│    Box 2: "diamond" at (200, 100), size 60×60, conf 0.91               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STAGE 2: OCR                                    │
│                                                                         │
│  Model: trocr-small (TrOCR, ONNX)                                      │
│  Input: Cropped image region for each text-containing shape            │
│  Output: Extracted text string                                         │
│                                                                         │
│  Process:                                                               │
│    1. For each shape that can contain text:                            │
│       - Crop the bounding box region from original image               │
│       - Encode as PNG → base64                                         │
│       - Run TrOCR inference                                            │
│       - Apply dictionary correction                                     │
│                                                                         │
│  Text containers:                                                       │
│    rectangle, rounded_rectangle, oval, circle, diamond,                │
│    hexagon, parallelogram, sticky_note, text_label, cloud,             │
│    cylinder, square, ellipse, document_shape, arrow_box                │
│                                                                         │
│  Non-text (skipped):                                                   │
│    solid_arrow, dashed_arrow, dotted_line, curved_arrow, etc.          │
│                                                                         │
│  Example:                                                               │
│    Crop rectangle at (50,30) → TrOCR → "Usr Auth" → Dict → "User Auth" │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 2b: DICTIONARY CORRECTION                    │
│                                                                         │
│  Pure Rust, no ML                                                       │
│                                                                         │
│  Process:                                                               │
│    1. Split OCR output into words                                      │
│    2. For each word, find closest match in dictionary                  │
│    3. If Levenshtein distance is small enough, replace                 │
│                                                                         │
│  Example:                                                               │
│    "Authntication" → distance 2 from "Authentication" → corrected      │
│    "Databse" → distance 1 from "Database" → corrected                  │
│    "XYZ123" → no close match → kept as-is                              │
│                                                                         │
│  Dictionary sources:                                                    │
│    - Common diagram terms (Start, End, Process, Database, etc.)        │
│    - User corrections (learned over time)                              │
│    - Context terms (from code, Jira, Slack)                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 3: LAYOUT ANALYSIS                           │
│                                                                         │
│  Pure Rust, no ML                                                       │
│                                                                         │
│  Process:                                                               │
│    1. Identify arrows vs shapes                                        │
│    2. For each arrow:                                                  │
│       - Find arrow start point (left edge center)                      │
│       - Find arrow end point (right edge center)                       │
│       - Find closest shape to start → "source"                         │
│       - Find closest shape to end → "target"                           │
│       - Record: source.connects_to.push(target)                        │
│    3. Infer diagram type from shapes present                           │
│                                                                         │
│  Arrow matching:                                                        │
│                                                                         │
│    ┌──────────┐                    ┌──────────┐                        │
│    │  User    │ ───────────────▶   │  Server  │                        │
│    └──────────┘                    └──────────┘                        │
│          ▲                              ▲                               │
│          │                              │                               │
│     start point                    end point                           │
│     closest to "User"              closest to "Server"                 │
│                                                                         │
│  Diagram type inference:                                                │
│    - stick_figure present → sequenceDiagram                            │
│    - cylinder/database_icon → erDiagram                                │
│    - diamond present → flowchart                                       │
│    - oval only → stateDiagram                                          │
│    - default → flowchart                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 4: MERMAID GENERATION                        │
│                                                                         │
│  Model: cyan-sketch (Phi-3 + LoRA, GGUF)                               │
│  Input: Structured prompt with shapes and connections                  │
│  Output: Valid Mermaid syntax                                          │
│                                                                         │
│  Prompt format:                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ <|user|>                                                         │   │
│  │ Convert this whiteboard to a Mermaid flowchart TD diagram.      │   │
│  │                                                                  │   │
│  │ Shapes detected:                                                 │   │
│  │ - Shape 0: rectangle containing "User"                          │   │
│  │ - Shape 2: diamond containing "Valid?"                          │   │
│  │ - Shape 3: rectangle containing "Process"                       │   │
│  │ - Shape 4: rectangle containing "End"                           │   │
│  │                                                                  │   │
│  │ Connections:                                                     │   │
│  │ - "User" → "Valid?"                                             │   │
│  │ - "Valid?" → "Process"                                          │   │
│  │ - "Valid?" → "End"                                              │   │
│  │                                                                  │   │
│  │ Generate valid Mermaid flowchart TD syntax:                     │   │
│  │ <|end|>                                                          │   │
│  │ <|assistant|>                                                    │   │
│  │ ```mermaid                                                       │   │
│  │ flowchart TD                                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LLM continues from there, generating:                                 │
│                                                                         │
│    flowchart TD                                                        │
│        A[User] --> B{Valid?}                                           │
│        B -->|Yes| C[Process]                                           │
│        B -->|No| D[End]                                                │
│        C --> D                                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      POST-PROCESSING                                    │
│                                                                         │
│  Extract mermaid code from LLM response:                               │
│    1. Look for ```mermaid ... ``` block                                │
│    2. If not found, look for any ``` ... ``` block                     │
│    3. If not found, return raw response                                │
│                                                                         │
│  Clean up:                                                              │
│    - Trim whitespace                                                   │
│    - Remove incomplete trailing lines                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
      │
      ▼
    Mermaid Code (ready to render)
```

## Code Walkthrough

### Entry Point: `process()`

```rust
pub fn process(&self, image_data: &[u8]) -> Result<PipelineResult> {
    // Load image once (reused for cropping)
    let img = image::load_from_memory(image_data)?;

    // Stage 1: YOLO detection
    let boxes = self.detect_shapes(image_data)?;

    // Stage 2: OCR with cropping
    let mut shapes = self.extract_text(&img, &boxes)?;

    // Stage 3: Layout analysis
    self.analyze_connections(&mut shapes, &boxes);
    let diagram_type = self.infer_diagram_type(&shapes);

    // Stage 4: LLM generation
    let mermaid = self.generate_mermaid(&shapes, &diagram_type)?;

    Ok(PipelineResult { shapes, mermaid, diagram_type, timing })
}
```

### Stage 1: `detect_shapes()`

```rust
fn detect_shapes(&self, image_data: &[u8]) -> Result<Vec<DetectedBox>> {
    // Encode image as base64 for JSON transport
    let input = InferenceInput::Image {
        data_base64: base64::engine::general_purpose::STANDARD.encode(image_data),
    };
    
    // Run YOLO inference
    let output = self.runtime.infer_sync(&self.yolo_model, input)?;

    match output {
        InferenceOutput::Boxes { detections } => Ok(detections),
        _ => Err(anyhow!("YOLO returned unexpected output type")),
    }
}
```

### Stage 2: `extract_text()` and `crop_and_ocr()`

```rust
fn extract_text(&self, img: &DynamicImage, boxes: &[DetectedBox]) -> Result<Vec<DetectedShape>> {
    let (img_w, img_h) = img.dimensions();
    
    for (id, box_) in boxes.iter().enumerate() {
        // Only OCR shapes that can contain text
        if Self::is_text_container(&box_.class_name) {
            // Crop and run OCR
            let text = self.crop_and_ocr(img, box_, img_w, img_h)?;
        }
        // ... build DetectedShape
    }
}

fn crop_and_ocr(&self, img: &DynamicImage, box_: &DetectedBox, ...) -> Result<Option<String>> {
    // 1. Clamp coordinates to image bounds
    let x = (box_.x.max(0.0) as u32).min(img_w - 1);
    let y = (box_.y.max(0.0) as u32).min(img_h - 1);
    // ...

    // 2. Skip tiny regions (< 10px)
    if w < 10 || h < 10 { return Ok(None); }

    // 3. Crop
    let cropped = img.crop_imm(x, y, w, h);

    // 4. Encode as PNG
    let mut png_bytes = Vec::new();
    cropped.write_to(&mut Cursor::new(&mut png_bytes), ImageFormat::Png)?;

    // 5. Run TrOCR
    let input = InferenceInput::Image {
        data_base64: base64::encode(&png_bytes),
    };
    let output = self.runtime.infer_sync(&self.trocr_model, input)?;

    // 6. Apply dictionary correction
    match output {
        InferenceOutput::Text { content } => {
            let corrected = self.dictionary.correct_phrase(&content);
            Ok(Some(corrected.corrected))
        }
        _ => Ok(None),
    }
}
```

### Stage 3: `analyze_connections()`

```rust
fn analyze_connections(&self, shapes: &mut [DetectedShape], boxes: &[DetectedBox]) {
    // Separate arrows from shapes
    let arrow_indices = boxes.iter().enumerate()
        .filter(|(_, b)| Self::is_arrow(&b.class_name))
        .map(|(i, _)| i).collect();
    
    let non_arrow_indices = /* ... opposite ... */;

    // For each arrow, find what it connects
    for arrow_idx in arrow_indices {
        let arrow = &boxes[arrow_idx];
        
        // Arrow start = left edge center
        let (start_x, start_y) = (arrow.x, arrow.y + arrow.height / 2.0);
        // Arrow end = right edge center
        let (end_x, end_y) = (arrow.x + arrow.width, arrow.y + arrow.height / 2.0);

        // Find closest shape to each endpoint
        for &shape_idx in &non_arrow_indices {
            let center = /* shape center */;
            let dist_to_start = distance(start_x, start_y, center.0, center.1);
            let dist_to_end = distance(end_x, end_y, center.0, center.1);
            
            // Track closest (within 150px max)
            // ...
        }

        // Record connection: source → target
        shapes[start_shape].connects_to.push(end_shape);
    }
}
```

### Stage 4: `build_prompt()` and `generate_mermaid()`

```rust
fn build_prompt(&self, shapes: &[DetectedShape], diagram_type: &DiagramType) -> String {
    let mut prompt = String::from("<|user|>\n");
    prompt.push_str("Convert this whiteboard to a Mermaid ... diagram.\n\n");
    
    // List shapes
    prompt.push_str("Shapes detected:\n");
    for shape in shapes.iter().filter(|s| !Self::is_arrow(&s.shape_type)) {
        let text = shape.text.as_deref().unwrap_or("[no text]");
        prompt.push_str(&format!("- Shape {}: {} containing \"{}\"\n", 
            shape.id, shape.shape_type, text));
    }
    
    // List connections
    prompt.push_str("\nConnections:\n");
    for shape in shapes {
        for &target_id in &shape.connects_to {
            // "User" → "Server"
        }
    }
    
    // Prime the LLM to continue
    prompt.push_str("<|end|>\n<|assistant|>\n```mermaid\nflowchart TD\n");
    
    prompt
}

fn generate_mermaid(&self, shapes: &[DetectedShape], diagram_type: &DiagramType) -> Result<String> {
    let prompt = self.build_prompt(shapes, diagram_type);
    
    let input = InferenceInput::Text { prompt };
    let output = self.runtime.infer_sync(&self.phi_model, input)?;
    
    match output {
        InferenceOutput::Text { content } => Ok(Self::extract_mermaid_code(&content)),
        _ => Err(anyhow!("Phi returned unexpected output type")),
    }
}
```

## Timing

Typical timing on M3 Max:

| Stage | Time |
|-------|------|
| YOLO detection | ~5-10ms |
| TrOCR (per shape) | ~50-100ms |
| Layout analysis | <1ms |
| Phi generation | ~500-2000ms |
| **Total** | **~1-3 seconds** |

## Limitations

1. **Arrow direction**: Assumes left→right. Vertical arrows may misconnect.
2. **Overlapping shapes**: May confuse detection.
3. **Small text**: OCR struggles with tiny or blurry text.
4. **Complex diagrams**: >15 shapes may overwhelm the prompt context.
