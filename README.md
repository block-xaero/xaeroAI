# XaeroAI

Fine-tuned AI models for software design analysis, integrated with the Cyan ecosystem.

## What is this?

XaeroAI provides the AI backbone for Cyan's design-first collaboration tools:
- **Design Analyst**: Analyze project health from Jira/Slack/GitHub data
- **Mermaid Generation**: Create sequence diagrams, flowcharts, class diagrams
- **Pattern Detection**: Identify GoF, enterprise, and Rust design patterns

## Current Model

| Model | Base | Size | Format |
|-------|------|------|--------|
| Cyan Lens v4 | Phi-3-mini-4k-instruct | ~2GB | GGUF Q4_K_M |
| Cyan Lens Segmentation | yolox | TBD | GGUF TBD |

**HuggingFace**: [blockxaero/cyan-lens](https://huggingface.co/blockxaero/cyan-lens)

## Capabilities
(Cyan Lens v4 only, stay tuned for Lens Segmentation capabilities).
### Project Health Analysis
```
Input: <context>Ticket: AUTH-101, Days open: 45, Slack mentions: 67, PRs: 0</context>

Output: {
  "status": "critical",
  "issues": ["No PRs after 45 days", "High team concern (67 mentions)"],
  "recommendations": ["Assign owner", "Break into smaller tasks"]
}
```

### Mermaid Diagram Generation
```
Input: Create a Mermaid sequenceDiagram for OAuth2 login

Output:
sequenceDiagram
    participant User
    participant App
    participant AuthServer
    User->>App: Click Login
    App->>AuthServer: Auth Request
    AuthServer-->>App: Token
```

### Diagram Support

| Type | Status |
|------|--------|
| sequenceDiagram | ✅ Works |
| flowchart | ✅ Works |
| classDiagram | ⚠️ Partial |
| erDiagram | ❌ Unreliable |
| stateDiagram | ❌ Unreliable |

## Project Structure

```
xaeroai/
├── src/lib.rs              # Rust types for Cyan integration
├── adapters/
│   └── design-analyst-v4/  # MLX LoRA adapter
├── models/                 # GGUF models (gitignored, download from HF)
├── data/
│   └── mlx_final/          # Training data
├── scripts/
│   ├── train.py            # Train/continue training
│   ├── export.py           # Merge LoRA + GGUF conversion
│   ├── test.py             # Test model outputs
│   └── generate_data.py    # Generate training data
└── lora_config.yaml        # Training configuration
```

## Usage

### Quick Test (llama-cli)
```bash
llama-cli -m models/cyan-lens-q4.gguf \
    -p "<|user|>Create a Mermaid sequenceDiagram for user login<|end|><|assistant|>" \
    -n 400
```

### Ollama
```bash
ollama create cyan-lens -f Modelfile
ollama run cyan-lens "Analyze: 45 days open, 67 Slack mentions, 0 PRs"
```

### MLX (Apple Silicon)
```bash
python -m mlx_lm generate \
    --model microsoft/Phi-3-mini-4k-instruct \
    --adapter-path adapters/design-analyst-v4 \
    --max-tokens 400 \
    --prompt "Create a Mermaid sequenceDiagram for OAuth2"
```

## Training Pipeline

```bash
# 1. Train new adapter version
python scripts/train.py --config lora_config.yaml

# 2. Test with MLX
python scripts/test.py --use-mlx

# 3. Export to GGUF
python scripts/export.py --adapter adapters/design-analyst-v4

# 4. Test GGUF
python scripts/test.py --model models/design-analyst-v4-q4.gguf
```

## Rust Integration (WIP)

The `src/lib.rs` defines types for Cyan backend integration:

```rust
use xaeroai::{DesignAnalyst, ModelConfig, AnalysisSource};

// Load model
let config = ModelConfig::default();
let mut analyst = DesignAnalyst::new(config);
analyst.load().await?;

// Analyze
let result = analyst.analyze(AnalysisSource::RawText {
    content: "<context>Ticket: AUTH-101...</context>".into(),
    language: None,
}).await?;
```

**Status**: Types defined, inference not yet implemented. Will use `llama-cpp-2` crate for GGUF loading.

## Future Integration

XaeroAI will integrate with:
- **Cyan Backend** (Swift) - Design analysis in whiteboard app
- **XaeroFlux** - Event-driven inference requests
- **XaeroID** - Authenticated model access

## Development

### Requirements
- Python 3.11+
- MLX (Apple Silicon) or PyTorch
- llama.cpp (for GGUF)
- Rust (for lib)

### Install
```bash
pip install mlx-lm transformers peft safetensors
npm install -g @mermaid-js/mermaid-cli  # For validation
```

## License

Business Source License 1.1
