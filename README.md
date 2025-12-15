# XaeroAI

AI runtime for Cyan whiteboard-to-diagram pipeline.

## Pipeline

```
Whiteboard Photo → YOLO (shapes) → PaddleOCR (text) → Dictionary (correct) → Phi-3 (mermaid)
```

## Models

| Model | Purpose | Size | Format |
|-------|---------|------|--------|
| cyan-sketch | Shape detection | 25MB | ONNX |
| paddleocr | Text recognition | 7.5MB | ONNX |
| cyan-lens | Mermaid generation | 2GB | GGUF Q4 |

Models: `scripts/models/`

## Usage

```bash
cargo run --release --bin test_pipeline -- \
  --models-dir scripts/models \
  --image whiteboard.jpg \
  --verbose
```

## Output

```
Detected Shapes (14):
  [0] rectangle  conf=0.87  "START"
  [1] diamond    conf=0.70  "Valid?"
  [2] rectangle  conf=0.84  "Process"
  ...

Generated Mermaid:
flowchart TD
  A[START] --> B{Valid?}
  B -->|Yes| C[Process]
```

## Project Structure

```
xaeroai/
├── src/
│   ├── lib.rs           # Library exports
│   ├── pipeline.rs      # Main pipeline orchestration
│   ├── runtime.rs       # ONNX/GGUF model loading
│   ├── arrow_detector.rs # Line/arrow detection
│   ├── dictionary.rs    # OCR correction
│   └── skill.rs         # Model metadata
├── scripts/models/
│   ├── whiteboard-detector/
│   ├── paddleocr/
│   └── cyan-lens/
└── config/
    └── ocr_dictionary.json
```

## Configuration

OCR dictionary in `config/ocr_dictionary.json`:

```json
{
  "min_length": 3,
  "max_distance_short": 1,
  "max_distance_long": 2,
  "terms": {
    "flow_control": ["START", "END", "YES", "NO"],
    "components": ["USER", "SERVER", "API", "DATABASE"]
  }
}
```

## Dependencies

- `ort` - ONNX Runtime
- `llama-cpp-2` - GGUF inference
- `image` / `imageproc` - Image processing

## License

Business Source License 1.1
