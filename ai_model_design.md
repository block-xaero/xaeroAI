# Cyan AI Model Registry - Hacking Context Document

## Project Overview

**Cyan** is a decentralized collaborative whiteboard application combining Discord-like functionality with Pinterest-style workspaces. It's built for offline-first, cloudless operation.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  UI Layer (Thin)                                                 │
│  ├── Swift (macOS / iOS / iPad) - primary                       │
│  └── Flutter (Android / Windows / Linux) - planned              │
│                                                                  │
│  Responsibilities: Camera, display, user interaction only       │
└─────────────────────────────────────────────────────────────────┘
                              │ FFI (C ABI)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Rust Backend (ALL heavy lifting)                               │
│  ├── Tokio (async runtime)                                      │
│  ├── SQLite (local storage via rusqlite)                        │
│  ├── LMDB (high-performance KV, planned)                        │
│  ├── Iroh (P2P networking - QUIC + gossipsub)                   │
│  ├── Integration Bridge (Slack, Confluence, etc.)               │
│  └── AI Model Registry (NEW - this document)                    │
│                                                                  │
│  Crate structure:                                                │
│  ├── cyan-backend (main lib.rs, FFI exports)                    │
│  └── cyan-integrations (separate crate for external services)   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Offline-first**: Everything works without internet
2. **Cloudless**: No central servers, P2P only
3. **Zero-allocation hot paths**: Performance-critical code avoids heap
4. **POD-first**: Prefer `#[repr(C)]` structs with bytemuck
5. **Cache-aligned**: `#[repr(C, align(64))]` for concurrent data
6. **Thin UI**: Swift/Flutter only handles display, all logic in Rust

---

## Current Codebase Structure

### Rust Backend (lib.rs highlights)

```rust
// Database schema includes whiteboard_elements
CREATE TABLE whiteboard_elements (
    id TEXT PRIMARY KEY,
    board_id TEXT NOT NULL,
    element_type TEXT NOT NULL,  // "rectangle", "sticky_note", "arrow", etc.
    x REAL, y REAL, width REAL, height REAL,
    z_index INTEGER,
    style_json TEXT,
    content_json TEXT,
    created_at INTEGER,
    updated_at INTEGER
);

// Event system for Swift communication
pub enum SwiftEvent {
    Network(NetworkEvent),
    FileTree { ... },
    IntegrationEvent { ... },
    IntegrationGraph { ... },
}

// Two separate event buffers (recently fixed race condition)
pub struct CyanSystem {
    pub event_ffi_buffer: Arc<Mutex<VecDeque<String>>>,           // General events
    pub integration_event_buffer: Arc<Mutex<VecDeque<String>>>,   // Integration events
    // ...
}
```

### P2P Sync (XaeroFlux via Iroh)

- Files sync via content-addressed Blake3 hashes
- Group-based subscriptions: `group/{gid}/workspace/{wid}/object/{oid}`
- Merkle trees for efficient delta sync
- DHT discovery for peer finding

### Integration Bridge

Separate crate handling external services:
- Slack (OAuth, post messages)
- Confluence (create/update pages)
- More planned

---

## AI Vision: Decentralized Model Registry

### The Idea

Users can drop AI model files into a Cyan workspace, and they become available as "chatbots" or "skills" - locally loaded, P2P synced (except LoRA weights which stay local).

### Use Cases

1. **Whiteboard OCR**: Snap photo → detect objects → extract text → generate summary
2. **Custom Chatbots**: Drop a fine-tuned LLM, it becomes a chat participant
3. **Domain Skills**: Specialized models for code review, design critique, etc.
4. **Shared Team Models**: Sync trained models across team via P2P

### File Convention

```
workspace/
├── models/
│   ├── whiteboard-ocr.safetensors      # Weights (syncs P2P)
│   ├── whiteboard-ocr.json             # Config (syncs P2P)
│   ├── whiteboard-ocr.skill.md         # Skill definition (syncs P2P)
│   └── whiteboard-ocr.lora.safetensors # LoRA weights (LOCAL ONLY)
```

### What Syncs vs What Stays Local

| File Pattern | Syncs P2P | Reason |
|--------------|-----------|--------|
| `*.safetensors` | ✅ Yes | Base model, shared |
| `*.json` | ✅ Yes | Config, shared |
| `*.skill.md` | ✅ Yes | Capabilities, shared |
| `*.lora.safetensors` | ❌ No | User-specific personalization |

---

## Model Config Format

### {model-name}.json

```json
{
  "name": "whiteboard-ocr",
  "version": "1.0.0",
  "architecture": "trocr-small",
  "type": "ocr",
  "input": {
    "type": "image",
    "formats": ["png", "jpeg", "webp"],
    "max_dimensions": [1024, 1024]
  },
  "output": {
    "type": "text"
  },
  "requirements": {
    "min_memory_mb": 256,
    "gpu_preferred": true
  },
  "metadata": {
    "author": "rick",
    "license": "MIT",
    "trained_on": "custom whiteboard dataset",
    "accuracy": "96.5% on test set"
  }
}
```

### Supported Architectures (Initial)

| Architecture | Type | Use Case | Size |
|--------------|------|----------|------|
| `yolox-nano` | detection | Object detection | ~4MB |
| `yolox-small` | detection | Object detection | ~9MB |
| `trocr-small` | ocr | Handwriting recognition | ~80MB |
| `trocr-base` | ocr | Handwriting recognition | ~330MB |
| `phi3-mini` | llm | Text generation/chat | ~2GB Q4 |
| `mistral-7b` | llm | Text generation/chat | ~4GB Q4 |

---

## SKILL.md Format (Anthropic-Inspired)

```markdown
# Whiteboard OCR

## Description
Recognizes handwritten text from whiteboard photos. Optimized for software 
design diagrams, sticky notes, and architectural sketches.

## Capabilities
- Handwritten text recognition
- Multiple text regions per image
- Confidence scores per recognition

## Input
- Type: Image (PNG, JPEG)
- Recommended: Clear, well-lit photos
- Max size: 1024x1024 (will be resized)

## Output
- Type: Text
- Format: JSON array of recognized text blocks with bounding boxes

## Triggers
Invoke this model when:
- User uploads/shares a whiteboard photo
- User says "read this", "OCR this", "what does this say"
- Image contains handwritten content

## Example

Input: [photo of whiteboard with "User → Auth → DB" written]

Output:
```json
{
  "texts": [
    {"text": "User", "bbox": [10, 20, 100, 50], "confidence": 0.95},
    {"text": "Auth", "bbox": [150, 20, 100, 50], "confidence": 0.92},
    {"text": "DB", "bbox": [300, 20, 80, 50], "confidence": 0.97}
  ]
}
```

## Limitations
- English only (for now)
- Struggles with very cursive handwriting
- Needs reasonable lighting

## Training
- Base: Microsoft TrOCR-small
- Fine-tuned on: 100 custom whiteboard photos
- LoRA rank: 8
```

---

## Rust Implementation Plan

### New Module Structure

```
cyan-backend/
├── src/
│   ├── lib.rs                      # Existing, add model FFI exports
│   ├── models/
│   │   ├── mod.rs                  # Model registry
│   │   ├── registry.rs             # LoadedModel, ModelConfig
│   │   ├── loader.rs               # SafeTensors loading
│   │   ├── skill.rs                # SKILL.md parser
│   │   └── backends/
│   │       ├── mod.rs
│   │       ├── yolox.rs            # YOLOX detection backend
│   │       ├── trocr.rs            # TrOCR OCR backend
│   │       └── llm.rs              # LLM backend (Phi3, Mistral)
│   │
│   └── vision/                     # Higher-level vision pipeline
│       ├── mod.rs
│       ├── pipeline.rs             # Detect → OCR → Summary
│       └── whiteboard.rs           # Whiteboard-specific logic
```

### Core Types

```rust
// src/models/registry.rs

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub static MODEL_REGISTRY: OnceLock<Arc<Mutex<ModelRegistry>>> = OnceLock::new();

pub struct ModelRegistry {
    models: HashMap<String, LoadedModel>,
}

pub struct LoadedModel {
    pub id: String,
    pub config: ModelConfig,
    pub skill: SkillDefinition,
    pub backend: Box<dyn ModelBackend + Send + Sync>,
    pub loaded_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub version: String,
    pub architecture: String,
    #[serde(rename = "type")]
    pub model_type: ModelType,
    pub input: InputSpec,
    pub output: OutputSpec,
    #[serde(default)]
    pub requirements: Requirements,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Detection,
    Ocr,
    Llm,
    Classification,
    Embedding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSpec {
    #[serde(rename = "type")]
    pub input_type: String,  // "image", "text", "audio"
    #[serde(default)]
    pub formats: Vec<String>,
    #[serde(default)]
    pub max_dimensions: Option<[u32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSpec {
    #[serde(rename = "type")]
    pub output_type: String,  // "text", "detections", "embeddings"
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Requirements {
    #[serde(default)]
    pub min_memory_mb: u32,
    #[serde(default)]
    pub gpu_preferred: bool,
}
```

### Model Backend Trait

```rust
// src/models/backends/mod.rs

use burn::tensor::Tensor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelInput {
    Image(Vec<u8>),
    Text(String),
    Tokens(Vec<u32>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelOutput {
    Text(String),
    Detections(Vec<Detection>),
    Embeddings(Vec<f32>),
    Json(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

pub trait ModelBackend: Send + Sync {
    fn invoke(&self, input: ModelInput) -> Result<ModelOutput, ModelError>;
    fn model_type(&self) -> ModelType;
    fn unload(&mut self) -> Result<(), ModelError>;
}
```

### SKILL.md Parser

```rust
// src/models/skill.rs

#[derive(Debug, Clone, Default)]
pub struct SkillDefinition {
    pub name: String,
    pub description: String,
    pub capabilities: Vec<String>,
    pub triggers: Vec<String>,
    pub input_description: String,
    pub output_description: String,
    pub examples: Vec<SkillExample>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SkillExample {
    pub input: String,
    pub output: String,
}

pub fn parse_skill_md(content: &str) -> Result<SkillDefinition, ParseError> {
    let mut skill = SkillDefinition::default();
    let mut current_section = String::new();
    let mut current_content = String::new();
    
    for line in content.lines() {
        if line.starts_with("# ") {
            skill.name = line.trim_start_matches("# ").to_string();
        } else if line.starts_with("## ") {
            // Save previous section
            save_section(&mut skill, &current_section, &current_content);
            current_section = line.trim_start_matches("## ").to_lowercase();
            current_content.clear();
        } else {
            current_content.push_str(line);
            current_content.push('\n');
        }
    }
    
    // Save last section
    save_section(&mut skill, &current_section, &current_content);
    
    Ok(skill)
}

fn save_section(skill: &mut SkillDefinition, section: &str, content: &str) {
    let content = content.trim();
    match section {
        "description" => skill.description = content.to_string(),
        "capabilities" => {
            skill.capabilities = content.lines()
                .filter(|l| l.starts_with("- "))
                .map(|l| l.trim_start_matches("- ").to_string())
                .collect();
        }
        "triggers" => {
            skill.triggers = content.lines()
                .filter(|l| l.starts_with("- "))
                .map(|l| l.trim_start_matches("- ").to_string())
                .collect();
        }
        "limitations" => {
            skill.limitations = content.lines()
                .filter(|l| l.starts_with("- "))
                .map(|l| l.trim_start_matches("- ").to_string())
                .collect();
        }
        _ => {}
    }
}
```

### FFI Exports

```rust
// src/lib.rs - add these exports

/// Initialize the model registry
#[unsafe(no_mangle)]
pub extern "C" fn cyan_init_model_registry() -> bool {
    MODEL_REGISTRY.set(Arc::new(Mutex::new(ModelRegistry::new()))).is_ok()
}

/// Load a model from safetensors file
/// Expects companion {name}.json and optional {name}.skill.md
#[unsafe(no_mangle)]
pub extern "C" fn cyan_load_model(safetensors_path: *const c_char) -> *mut c_char {
    let path = unsafe { CStr::from_ptr(safetensors_path) }.to_str().unwrap();
    
    let config_path = path.replace(".safetensors", ".json");
    let skill_path = path.replace(".safetensors", ".skill.md");
    
    let result = load_model_from_files(path, &config_path, &skill_path);
    
    let response = match result {
        Ok(model_id) => json!({
            "success": true,
            "model_id": model_id,
        }),
        Err(e) => json!({
            "success": false,
            "error": e.to_string(),
        }),
    };
    
    CString::new(response.to_string()).unwrap().into_raw()
}

/// List all loaded models
#[unsafe(no_mangle)]
pub extern "C" fn cyan_list_models() -> *mut c_char {
    let registry = MODEL_REGISTRY.get().unwrap().lock().unwrap();
    
    let models: Vec<_> = registry.models.values()
        .map(|m| json!({
            "id": m.id,
            "name": m.config.name,
            "type": m.config.model_type,
            "architecture": m.config.architecture,
            "capabilities": m.skill.capabilities,
        }))
        .collect();
    
    CString::new(serde_json::to_string(&models).unwrap()).unwrap().into_raw()
}

/// Invoke a model
#[unsafe(no_mangle)]
pub extern "C" fn cyan_invoke_model(
    model_id: *const c_char,
    input_json: *const c_char,
) -> *mut c_char {
    let model_id = unsafe { CStr::from_ptr(model_id) }.to_str().unwrap();
    let input_json = unsafe { CStr::from_ptr(input_json) }.to_str().unwrap();
    
    let input: ModelInput = serde_json::from_str(input_json).unwrap();
    
    let registry = MODEL_REGISTRY.get().unwrap().lock().unwrap();
    let model = registry.models.get(model_id);
    
    let result = match model {
        Some(m) => m.backend.invoke(input),
        None => Err(ModelError::NotFound(model_id.to_string())),
    };
    
    let response = match result {
        Ok(output) => json!({
            "success": true,
            "output": output,
        }),
        Err(e) => json!({
            "success": false,
            "error": e.to_string(),
        }),
    };
    
    CString::new(response.to_string()).unwrap().into_raw()
}

/// Unload a model (free memory)
#[unsafe(no_mangle)]
pub extern "C" fn cyan_unload_model(model_id: *const c_char) -> bool {
    let model_id = unsafe { CStr::from_ptr(model_id) }.to_str().unwrap();
    let mut registry = MODEL_REGISTRY.get().unwrap().lock().unwrap();
    registry.models.remove(model_id).is_some()
}

/// Load LoRA weights onto an existing model (local only, never synced)
#[unsafe(no_mangle)]
pub extern "C" fn cyan_load_lora(
    model_id: *const c_char,
    lora_path: *const c_char,
) -> bool {
    // LoRA loading - enhances base model with user-specific weights
    // These .lora.safetensors files are LOCAL ONLY
    todo!()
}

/// Check if a file looks like a loadable model
#[unsafe(no_mangle)]
pub extern "C" fn cyan_is_model_file(path: *const c_char) -> bool {
    let path = unsafe { CStr::from_ptr(path) }.to_str().unwrap();
    
    if !path.ends_with(".safetensors") {
        return false;
    }
    
    // Check for companion config file
    let config_path = path.replace(".safetensors", ".json");
    Path::new(&config_path).exists()
}
```

### Model Loader

```rust
// src/models/loader.rs

use burn::record::Recorder;
use burn_import::safetensors::SafeTensorRecorder;

pub fn load_model_from_files(
    weights_path: &str,
    config_path: &str,
    skill_path: &str,
) -> Result<String, ModelError> {
    // 1. Load config
    let config: ModelConfig = serde_json::from_reader(
        File::open(config_path).map_err(|e| ModelError::ConfigNotFound(e.to_string()))?
    )?;
    
    // 2. Load skill definition (optional)
    let skill = if Path::new(skill_path).exists() {
        let content = std::fs::read_to_string(skill_path)?;
        parse_skill_md(&content)?
    } else {
        SkillDefinition::default_for(&config)
    };
    
    // 3. Load weights based on architecture
    let device = burn::backend::wgpu::WgpuDevice::default();
    
    let backend: Box<dyn ModelBackend + Send + Sync> = match config.architecture.as_str() {
        "yolox-nano" | "yolox-small" | "yolox-medium" => {
            Box::new(YoloxBackend::load(weights_path, &config, &device)?)
        }
        "trocr-small" | "trocr-base" => {
            Box::new(TrOcrBackend::load(weights_path, &config, &device)?)
        }
        "phi3-mini" | "phi3-small" => {
            Box::new(LlmBackend::load(weights_path, &config, &device)?)
        }
        arch => {
            return Err(ModelError::UnsupportedArchitecture(arch.to_string()));
        }
    };
    
    // 4. Generate model ID
    let model_id = format!("{}-{}", config.name, &config.version);
    
    // 5. Register
    let mut registry = MODEL_REGISTRY.get().unwrap().lock().unwrap();
    registry.models.insert(model_id.clone(), LoadedModel {
        id: model_id.clone(),
        config,
        skill,
        backend,
        loaded_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64,
    });
    
    Ok(model_id)
}
```

---

## Swift UI Integration

### File Panel Detection

```swift
// In FileTreeView or FilesPanel

func isModelFile(_ node: FileNode) -> Bool {
    guard node.name.hasSuffix(".safetensors") else { return false }
    
    // Check for companion .json
    let configName = node.name.replacingOccurrences(of: ".safetensors", with: ".json")
    return siblings.contains { $0.name == configName }
}

// Show "Load as Chatbot" button for model files
if isModelFile(node) {
    Button("Load as Chatbot") {
        loadModel(node.path)
    }
}
```

### Model Loading

```swift
// CyanBridge.swift

func loadModel(path: String) -> ModelLoadResult {
    let result = cyan_load_model(path.cString(using: .utf8))
    defer { cyan_free_string(result) }
    
    let json = String(cString: result!)
    return try! JSONDecoder().decode(ModelLoadResult.self, from: json.data(using: .utf8)!)
}

func listModels() -> [LoadedModel] {
    let result = cyan_list_models()
    defer { cyan_free_string(result) }
    
    let json = String(cString: result!)
    return try! JSONDecoder().decode([LoadedModel].self, from: json.data(using: .utf8)!)
}

func invokeModel(modelId: String, input: ModelInput) -> ModelOutput {
    let inputJson = try! JSONEncoder().encode(input)
    let result = cyan_invoke_model(
        modelId.cString(using: .utf8),
        String(data: inputJson, encoding: .utf8)!.cString(using: .utf8)
    )
    defer { cyan_free_string(result) }
    
    let json = String(cString: result!)
    return try! JSONDecoder().decode(ModelOutput.self, from: json.data(using: .utf8)!)
}
```

### Chat Integration

```swift
// ChatViewModel.swift

class ChatViewModel: ObservableObject {
    @Published var availableModels: [LoadedModel] = []
    @Published var activeModel: LoadedModel?
    
    func refreshModels() {
        availableModels = CyanBridge.listModels()
    }
    
    func sendMessage(_ text: String, image: Data? = nil) async {
        guard let model = activeModel else { return }
        
        let input: ModelInput
        if let imageData = image {
            input = .image(imageData)
        } else {
            input = .text(text)
        }
        
        let output = await CyanBridge.invokeModel(modelId: model.id, input: input)
        
        // Handle output based on type
        switch output {
        case .text(let response):
            appendMessage(.assistant(response))
        case .detections(let dets):
            appendMessage(.detections(dets))
        // ...
        }
    }
}
```

---

## Cargo.toml Additions

```toml
[dependencies]
# Existing deps...
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# New for AI models
burn = { version = "0.19", features = ["wgpu", "ndarray"] }
burn-import = { version = "0.19", features = ["onnx"] }
safetensors = "0.4"
image = "0.25"
imageproc = "0.24"

# For LLM support (optional, feature-gated)
# llama-cpp-rs = { version = "0.3", optional = true }

[features]
default = ["vision"]
vision = []
llm = ["llama-cpp-rs"]
```

---

## Training Models (For Reference)

### YOLOX Training (PyTorch)

```bash
# Clone YOLOX
git clone https://github.com/Megvii-BaseDetection/YOLOX
cd YOLOX
pip install -e .

# Train on your data
python tools/train.py -f whiteboard_exp.py -d 1 -b 16 --fp16

# Export to ONNX
python tools/export_onnx.py --output yolox_whiteboard.onnx -f whiteboard_exp.py -c best_ckpt.pth
```

### TrOCR Fine-tuning (HuggingFace)

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")

# Fine-tune on your cropped text images
trainer = Seq2SeqTrainer(model=model, args=training_args, ...)
trainer.train()

# Save as SafeTensors
model.save_pretrained("./trocr-whiteboard", safe_serialization=True)
```

---

## Data Collection Checklist

### Photos Needed: ~100

- [ ] 40 full board shots (multiple objects)
- [ ] 30 close-ups (2-4 objects, clear text)
- [ ] 15 varied angles (perspective, shadows)
- [ ] 15 different conditions (lighting, board types)

### Annotation Tool

Use CVAT (free, web-based):
1. Create project with labels: `sticky_note`, `rectangle`, `arrow`, `text`, `circle`, `line`
2. Upload photos
3. Draw bounding boxes
4. Export as COCO JSON
5. Convert to YOLOX format

---

## Implementation Order

### Day 1: Scaffolding
- [ ] Create `src/models/` module structure
- [ ] Implement `ModelConfig` and `SkillDefinition` types
- [ ] Implement SKILL.md parser
- [ ] Add FFI exports (stubs)

### Day 2: Registry
- [ ] Implement `ModelRegistry`
- [ ] Implement `cyan_load_model` (config loading only, no weights yet)
- [ ] Implement `cyan_list_models`
- [ ] Test from Swift

### Day 3: Backend Trait
- [ ] Define `ModelBackend` trait
- [ ] Implement dummy backend for testing
- [ ] Wire up `cyan_invoke_model`

### Day 4: First Real Backend
- [ ] Add Burn dependencies
- [ ] Implement YOLOX backend (or TrOCR, whichever is simpler)
- [ ] Test with pre-trained weights

### Day 5: Swift UI
- [ ] Add "Load as Chatbot" to file panel
- [ ] Add model selector to chat view
- [ ] Wire up invocation

---

## License Summary

| Component | License | Commercial OK |
|-----------|---------|---------------|
| YOLOX | Apache 2.0 | ✅ |
| TrOCR | MIT | ✅ |
| Phi-3 | MIT | ✅ |
| Burn.rs | Apache 2.0 / MIT | ✅ |
| SafeTensors | Apache 2.0 | ✅ |

---

## Open Questions

1. **Model size limits?** Should we warn users about large models on mobile?

2. **Validation?** How paranoid about malicious models? SafeTensors is safer than pickle but still...

3. **GPU memory management?** Unload models when memory pressure?

4. **Model versioning?** What happens when user syncs updated model?

---

## Quick Reference

### FFI Functions

| Function | Purpose |
|----------|---------|
| `cyan_init_model_registry()` | Initialize registry |
| `cyan_load_model(path)` | Load model from .safetensors |
| `cyan_list_models()` | List loaded models |
| `cyan_invoke_model(id, input)` | Run inference |
| `cyan_unload_model(id)` | Free model memory |
| `cyan_load_lora(id, path)` | Load LoRA weights |
| `cyan_is_model_file(path)` | Check if file is loadable |

### File Patterns

| Pattern | Purpose | Syncs? |
|---------|---------|--------|
| `*.safetensors` | Model weights | ✅ |
| `*.json` | Model config | ✅ |
| `*.skill.md` | Skill definition | ✅ |
| `*.lora.safetensors` | LoRA weights | ❌ Local only |