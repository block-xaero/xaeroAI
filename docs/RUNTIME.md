# Runtime: Model Loading & Inference

## Overview

`runtime.rs` handles loading and running models. Two backends:

| Backend | Format | Library | Use Case |
|---------|--------|---------|----------|
| GGUF | `.gguf` | llama-cpp-2 | Text generation (Phi-3, Llama) |
| ONNX | `.onnx` | ort | Vision models (YOLO, TrOCR) |

## GGUF Inference (llama-cpp-2)

### What is GGUF?

GGUF = "GPT-Generated Unified Format". A binary format for LLM weights optimized for inference.

### How Text Generation Works

A language model predicts the **next token** given previous tokens. That's it.

```
Input:  "The cat sat on the"
         ↓
Model predicts probabilities for all 32,000 possible next tokens:
         ↓
Output: "mat" (73%), "floor" (15%), "roof" (8%), ...
         ↓
Sample one token (e.g., "mat")
         ↓
Repeat with: "The cat sat on the mat"
```

### Code Flow

```rust
fn infer_gguf(&self, model: &GgufModel, input: InferenceInput) -> Result<InferenceOutput>
```

#### Step 1: Extract Prompt

```rust
let prompt = match input {
    InferenceInput::Text { prompt } => prompt,
    // ...
};
```

The prompt might be:
```
"<|user|>\nCreate a flowchart for login\n<|end|>\n<|assistant|>\n"
```

#### Step 2: Create Context

```rust
let ctx_params = LlamaContextParams::default()
    .with_n_ctx(NonZeroU32::new(2048));

let mut ctx = model.model.new_context(get_llama_backend(), ctx_params)?;
```

**Context** = workspace for generation.
- `n_ctx = 2048` = can hold 2048 tokens max
- Includes the KV cache (stores attention states)

```
┌────────────────────────────────────────────────────────────────┐
│                        CONTEXT                                 │
├────────────────────────────────────────────────────────────────┤
│  n_ctx = 2048 tokens max                                       │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    KV CACHE                               │ │
│  │  Stores intermediate attention values so we don't        │ │
│  │  recompute them for every new token                      │ │
│  │                                                           │ │
│  │  Position 0: [attention values for token 0]              │ │
│  │  Position 1: [attention values for token 1]              │ │
│  │  Position 2: [attention values for token 2]              │ │
│  │  ...                                                      │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

#### Step 3: Tokenize

```rust
let tokens = model.model.str_to_token(&prompt, AddBos::Always)?;
```

Converts text to token IDs:

```
"Hello world" → [1, 15496, 995]
                 │    │      │
                 │    │      └── "world"
                 │    └── "Hello"
                 └── BOS (Beginning Of Sequence)
```

**Why tokens?**
- Fixed vocabulary (32,000 for Phi-3)
- Neural networks need numbers
- Handles any text, any language

#### Step 4: Create Batch

```rust
let mut batch = LlamaBatch::new(2048, 1);

for (i, token) in tokens.iter().enumerate() {
    let is_last = i == tokens.len() - 1;
    batch.add(*token, i as i32, &[0], is_last)?;
}
```

A **batch** groups tokens for efficient processing:

```
batch.add(token, position, sequence_ids, compute_logits)
            │        │           │              │
            │        │           │              └── Need output for this token?
            │        │           └── Which conversation (for multi-chat)
            │        └── Position in sequence
            └── The token ID
```

Only the **last** token needs `compute_logits = true` because we only predict from the last position.

#### Step 5: Decode (Forward Pass)

```rust
ctx.decode(&mut batch)?;
```

This is the **expensive** step. Runs the transformer:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER FORWARD PASS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input tokens: [1, 15496, 995]                                 │
│                                                                 │
│         ↓                                                       │
│  ┌─────────────────┐                                           │
│  │   Embedding     │  Token IDs → 3072-dim vectors             │
│  └────────┬────────┘                                           │
│           ↓                                                     │
│  ┌─────────────────┐                                           │
│  │   Layer 1       │  Self-attention + feed-forward            │
│  └────────┬────────┘                                           │
│           ↓                                                     │
│         ...          (32 layers for Phi-3)                     │
│           ↓                                                     │
│  ┌─────────────────┐                                           │
│  │   Layer 32      │                                           │
│  └────────┬────────┘                                           │
│           ↓                                                     │
│  ┌─────────────────┐                                           │
│  │   Output Head   │  3072-dim vector → 32,000 logits          │
│  └────────┬────────┘                                           │
│           ↓                                                     │
│  Logits: [0.1, -2.3, 5.7, -1.2, ...]  (32,000 values)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

After decode, the KV cache is filled. Future tokens are **fast** because we reuse cached values.

#### Step 6: Generation Loop

```rust
for _ in 0..max_tokens {
    // Get probabilities for next token
    let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
    let mut candidates_arr = LlamaTokenDataArray::from_iter(candidates, false);
```

`candidates_ith(last_position)` returns probability for each of 32,000 possible next tokens:

```
Token ID │ Logit │ Probability
─────────┼───────┼────────────
    0    │ -5.2  │   0.0001
    1    │ -4.8  │   0.0002
  ...    │  ...  │    ...
  8765   │  4.3  │   0.2341  ← "flow"
  8766   │  3.9  │   0.1523  ← "chart"
  ...    │  ...  │    ...
```

#### Step 7: Sample with Temperature

```rust
ctx.sample_temp(&mut candidates_arr, 0.7);
let new_token = ctx.sample_token(&mut candidates_arr);
```

**Temperature** controls randomness:

| Temperature | Behavior |
|-------------|----------|
| 0.0 | Always pick highest probability (deterministic, boring) |
| 0.7 | Balance creativity and coherence (default) |
| 1.5 | More random, can be weird |

#### Step 8: Check End & Continue

```rust
if model.model.is_eog_token(new_token) {
    break;  // Model says "I'm done"
}

output_tokens.push(new_token);

// Feed new token back
batch.clear();
batch.add(new_token, n_cur as i32, &[0], true)?;
ctx.decode(&mut batch)?;  // Fast! Uses KV cache
```

Each subsequent `decode()` is **fast** because it only processes 1 new token, reusing the KV cache.

#### Step 9: Detokenize

```rust
let output = output_tokens.iter()
    .filter_map(|t| model.model.token_to_str(*t, Special::Tokenize).ok())
    .collect::<String>();
```

Converts token IDs back to text:
```
[8765, 9310, 25, 13] → "flowchart TD\n"
```

---

## ONNX Inference (ort)

### What is ONNX?

ONNX = Open Neural Network Exchange. A portable format for ML models.

### YOLO Detection Flow

Unlike LLMs, YOLO runs **one forward pass** and outputs everything at once.

```rust
fn infer_onnx(&self, model: &OnnxModel, input: InferenceInput) -> Result<InferenceOutput>
```

#### Step 1: Decode Image

```rust
let image_bytes = base64::decode(&data_base64)?;
let img = image::load_from_memory(&image_bytes)?;
let (orig_width, orig_height) = img.dimensions();
```

#### Step 2: Preprocess

```rust
// Resize to YOLO's expected input size
let resized = img.resize_exact(640, 640, FilterType::Triangle);
let rgb = resized.to_rgb8();

// Create tensor [1, 3, 640, 640]
let mut input_tensor = Array::zeros(IxDyn(&[1, 3, 640, 640]));

// Normalize pixels: 0-255 → 0.0-1.0
for y in 0..640 {
    for x in 0..640 {
        let pixel = rgb.get_pixel(x, y);
        input_tensor[[0, 0, y, x]] = pixel[0] as f32 / 255.0;  // R
        input_tensor[[0, 1, y, x]] = pixel[1] as f32 / 255.0;  // G
        input_tensor[[0, 2, y, x]] = pixel[2] as f32 / 255.0;  // B
    }
}
```

Tensor shape `[1, 3, 640, 640]`:
- `1` = batch size (1 image)
- `3` = RGB channels
- `640, 640` = height, width

#### Step 3: Run Model

```rust
let outputs = model.session.run(ort::inputs![input_tensor]?)?;
```

**One call**. That's it. ONNX Runtime handles everything.

#### Step 4: Parse Output

YOLO output shape: `[1, 34, 8400]`
- `34` = 4 (box coords) + 30 (class scores)
- `8400` = number of detection candidates

```rust
for i in 0..8400 {
    // Box coordinates (in 640×640 space)
    let x_center = output[[0, 0, i]];
    let y_center = output[[0, 1, i]];
    let w = output[[0, 2, i]];
    let h = output[[0, 3, i]];

    // Find best class
    let mut best_class = 0;
    let mut best_conf = 0.0;
    for c in 0..30 {  // 30 classes
        let conf = output[[0, 4 + c, i]];
        if conf > best_conf {
            best_conf = conf;
            best_class = c;
        }
    }

    // Filter low confidence
    if best_conf > 0.25 {
        // Scale back to original image size
        let scale_x = orig_width / 640.0;
        let scale_y = orig_height / 640.0;
        
        detections.push(DetectedBox {
            class_id: best_class,
            x: (x_center - w/2.0) * scale_x,
            y: (y_center - h/2.0) * scale_y,
            width: w * scale_x,
            height: h * scale_y,
            confidence: best_conf,
        });
    }
}
```

#### Step 5: Non-Maximum Suppression (NMS)

YOLO often detects the same object multiple times:

```
Before NMS:
┌─────────────────────┐
│  ┌─────────────┐    │
│  │ ┌─────────┐ │    │  3 overlapping boxes
│  │ │  BOX    │ │    │  for same object
│  │ └─────────┘ │    │
│  └─────────────┘    │
└─────────────────────┘

After NMS:
┌─────────────────────┐
│      ┌─────┐        │  1 box (highest confidence)
│      │ BOX │        │
│      └─────┘        │
└─────────────────────┘
```

```rust
fn nms(&self, mut boxes: Vec<DetectedBox>, iou_threshold: f32) -> Vec<DetectedBox> {
    // Sort by confidence (highest first)
    boxes.sort_by(|a, b| b.confidence.cmp(&a.confidence));

    let mut keep = Vec::new();

    while !boxes.is_empty() {
        let best = boxes.remove(0);
        keep.push(best.clone());

        // Remove boxes that overlap too much
        boxes.retain(|b| {
            if b.class_id != best.class_id { return true; }
            compute_iou(&best, b) < iou_threshold
        });
    }

    keep
}
```

**IoU** (Intersection over Union):

```
┌─────────────────┐
│     Box A       │
│    ┌───────┼────┼───┐
│    │XXXXXXX│    │   │
│    │XXXXXXX│    │   │
└────┼───────┘    │   │
     │    Box B   │   │
     └────────────┴───┘

IoU = Area(XXXX) / Area(A ∪ B)

If IoU > 0.45 → same object → remove lower confidence
```

---

## LoRA Adapter Swapping

LoRA = Low-Rank Adaptation. Small adapter weights that modify a base model.

```
Base Model:     W₀  (3.8B parameters, frozen)
LoRA Adapter:   ΔW  (few MB, trained)

At inference:   W = W₀ + scale × ΔW
```

### Code

```rust
pub fn swap_lora(&mut self, base_model_id: &str, lora_path: &Path) -> Result<()> {
    let model = self.models.get_mut(base_model_id)?;

    match model {
        LoadedModel::Gguf(gguf) => {
            // llama-cpp-2 merges LoRA into the model
            gguf.model.lora_adapter_set(lora_path, 1.0)?;
            gguf.active_lora = Some(lora_path.to_string());
            Ok(())
        }
        LoadedModel::Onnx(_) => Err(anyhow!("Cannot apply LoRA to ONNX")),
    }
}
```

### Use Case

```rust
// Load base Phi-3 once
runtime.load_from_skill(&phi_skill, models_dir)?;

// Swap adapters for different tasks
runtime.swap_lora("phi-3", "adapters/mermaid-flowchart.safetensors")?;
// Generate flowcharts...

runtime.swap_lora("phi-3", "adapters/mermaid-sequence.safetensors")?;
// Generate sequence diagrams...
```

Swap time: ~10-50ms (just pointer math, no reloading).

---

## Comparison

| Aspect | GGUF (LLM) | ONNX (YOLO) |
|--------|------------|-------------|
| Input | Text | Image |
| Output | Text (streaming) | Boxes (all at once) |
| Forward passes | Many (1 per token) | One |
| Generation loop | Yes | No |
| Post-processing | Detokenize | NMS |
| Typical time | 500ms-2s | 2-10ms |
