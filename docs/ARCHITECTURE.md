# xaeroai Architecture

## Overview

xaeroai is the AI runtime for Cyan. It handles model loading, inference, and the whiteboard-to-mermaid pipeline.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              XAEROAI CRATE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │   lib.rs    │     │ runtime.rs  │     │ pipeline.rs │                   │
│  │             │     │             │     │             │                   │
│  │  FFI Bridge │────▶│  Inference  │────▶│  Pipeline   │                   │
│  │  Commands   │     │  GGUF/ONNX  │     │  YOLO→OCR→  │                   │
│  │  Events     │     │             │     │  LLM        │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│         │                   │                   │                           │
│         │            ┌──────┴──────┐            │                           │
│         │            ▼             ▼            │                           │
│         │     ┌──────────┐  ┌──────────┐       │                           │
│         │     │ llama-cpp│  │   ort    │       │                           │
│         │     │ (GGUF)   │  │  (ONNX)  │       │                           │
│         │     └──────────┘  └──────────┘       │                           │
│         │                                       │                           │
│         ▼                                       ▼                           │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │ registry.rs │     │ skill.rs    │     │dictionary.rs│                   │
│  │             │     │             │     │             │                   │
│  │  SQLite     │     │  SKILL.md   │     │  OCR Text   │                   │
│  │  Storage    │     │  Parser     │     │  Correction │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                           │
│  │correction.rs│                                                           │
│  │             │                                                           │
│  │  User       │                                                           │
│  │  Feedback   │                                                           │
│  └─────────────┘                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
xaeroai/
├── Cargo.toml           # Dependencies: llama-cpp-2, ort, image, base64, etc.
├── src/
│   ├── lib.rs           # FFI entry points, AISystem, Commands/Events
│   ├── runtime.rs       # Model loading and inference (GGUF + ONNX)
│   ├── pipeline.rs      # Whiteboard → Mermaid pipeline
│   ├── skill.rs         # SKILL.md manifest parsing
│   ├── registry.rs      # SQLite model_registry table
│   ├── correction.rs    # SQLite corrections table
│   └── dictionary.rs    # Fuzzy OCR text correction
├── scripts/
│   └── download_trocr.py
└── examples/
    ├── whiteboard-detector-SKILL.md
    ├── cyan-lens-SKILL.md
    └── cyan-sketch-SKILL.md
```

## Data Flow

### Swift → Rust → Swift

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            SWIFT APP                                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ModelActor                                                             │
│       │                                                                  │
│       │  1. Send command (JSON)                                          │
│       ▼                                                                  │
│   xaero_ai_command('{"type":"Infer","model_id":"yolo",...}')            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                            RUST (xaeroai)                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   lib.rs                                                                 │
│       │                                                                  │
│       │  2. Parse JSON → AICommand enum                                  │
│       │  3. AISystem.handle_command()                                    │
│       │  4. Call runtime.infer_sync()                                    │
│       │  5. Push AIEvent to queue                                        │
│       ▼                                                                  │
│   Event Queue: [InferenceComplete{...}]                                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                            SWIFT APP                                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ModelActor                                                             │
│       │                                                                  │
│       │  6. Poll for events                                              │
│       ▼                                                                  │
│   xaero_ai_poll_event() → '{"type":"InferenceComplete",...}'            │
│       │                                                                  │
│       │  7. Handle result, update UI                                     │
│       ▼                                                                  │
│   Display mermaid diagram                                                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Key Types

### Commands (Swift → Rust)

```rust
pub enum AICommand {
    RegisterModel { board_id, file_id, skill_md },
    UnregisterModel { model_id },
    LoadModel { model_id },
    UnloadModel { model_id },
    Infer { request_id, model_id, input_json },
    SwapLora { base_model_id, lora_model_id },
    LogCorrection { model_id, input_type, input_data, original, corrected },
    ListModels { board_id },
    ProcessWhiteboard { request_id, image_hash },
    // ...
}
```

### Events (Rust → Swift)

```rust
pub enum AIEvent {
    ModelLoaded { model_id, name },
    ModelUnloaded { model_id },
    InferenceComplete { request_id, model_id, output_json, latency_ms },
    InferenceError { request_id, model_id, error },
    CorrectionSaved { correction_id, model_id },
    WhiteboardProcessed { request_id, mermaid, diagram_type, shape_count, latency_ms },
    // ...
}
```

### Inference I/O

```rust
pub enum InferenceInput {
    Text { prompt: String },
    Image { data_base64: String },
    Json { data: serde_json::Value },
}

pub enum InferenceOutput {
    Text { content: String },
    Boxes { detections: Vec<DetectedBox> },
    Json { data: serde_json::Value },
}
```

## Model Types

| Type | Format | Runtime | Use Case |
|------|--------|---------|----------|
| GGUF | `.gguf` | llama-cpp-2 | LLMs (Phi-3, Llama) |
| ONNX | `.onnx` | ort | Vision (YOLO, TrOCR) |
| LoRA | `.safetensors` | llama-cpp-2 | Adapters for GGUF |

## Storage

### model_registry (SQLite)

```sql
CREATE TABLE model_registry (
    id TEXT PRIMARY KEY,
    board_id TEXT NOT NULL,      -- Parent board
    name TEXT NOT NULL,          -- "whiteboard-detector"
    version TEXT NOT NULL,       -- "0.1.0"
    kind TEXT NOT NULL,          -- "gguf", "onnx", "lora"
    capabilities TEXT NOT NULL,  -- JSON: ["image_to_boxes"]
    skill_md TEXT NOT NULL,      -- Full SKILL.md content
    model_hash TEXT NOT NULL,    -- Blake3 hash
    file_id TEXT,                -- Blob storage reference
    ...
);
```

### corrections (SQLite)

```sql
CREATE TABLE corrections (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    input_data TEXT NOT NULL,    -- Original prompt
    original TEXT NOT NULL,      -- Model output
    corrected TEXT NOT NULL,     -- User's fix
    synced INTEGER DEFAULT 0,    -- Sent to peers?
    drained INTEGER DEFAULT 0,   -- Used in training?
    ...
);
```
