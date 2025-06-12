# XaeroAI

Nano AI models for peer-to-peer distributed learning and inference.

## What is this?

XaeroAI provides small, efficient AI models that can:
- Run inference on mobile devices
- Learn from each other over a network
- Coordinate tasks without central servers

Each model is a network peer that discovers other models and collaborates on AI tasks.

## Models

We provide 4 nano models, each optimized for specific tasks:

| Model | Size | Purpose |
|-------|------|---------|
| Phi-2 Code | 63MB | Code completion and analysis |
| CodeGen 350M | 60MB | Code generation |
| TrOCR Small | 56MB | Handwritten text recognition |
| TrOCR Base | 47MB | Printed text recognition |

Total: ~226MB for all models.

## How it works

### Model Storage
Models are compressed and split into 25MB chunks for efficient distribution:
```
models/nano/
├── phi2_ultra_code_nano_part_000     # 25MB chunk
├── phi2_ultra_code_nano_part_001     # 25MB chunk  
├── phi2_ultra_code_nano_manifest.txt # Assembly instructions
└── ... (other models)
```

### Memory Loading
Models use memory-mapped loading via Candle:
- Load only the parts needed for current tasks
- Efficient memory usage on mobile devices
- Stream model weights as needed

### Peer-to-Peer Learning
Models connect through XaeroFlux subjects:
- Discover other models in the network
- Share learning updates via CRDT operations
- Coordinate multi-model tasks
- Learn from workspace events (whiteboard, chat, etc.)

## Architecture

```
AI Model Peer
├── Local inference (Candle + mmap)
├── Network coordination (XaeroFlux subjects)
├── Learning from events (CRDT operations)
└── Task collaboration (gossipsub)
```

Each model subscribes to relevant subjects:
- `workspace/ai/discovery` - Find other models
- `workspace/ai/tasks` - Coordinate inference requests
- `workspace/ai/learning` - Share parameter updates
- `workspace/{id}/data` - Learn from workspace activity

## Usage

```rust
use xaeroai::{XaeroAIModel, XaeroAISubject};

// Load a nano model
let model = XaeroAIModel::load("phi2_ultra_code_nano")?;

// Connect to network
let subject = XaeroAISubject::new(model.id)?;
subject.subscribe_to_discovery();

// The model automatically:
// - Announces itself to other peers
// - Responds to inference requests
// - Learns from workspace events
// - Coordinates with other models
```

## Integration

XaeroAI integrates with:
- **XaeroFlux** - Event streaming and CRDT operations
- **XaeroID** - Peer identity and authentication
- **Candle** - Efficient model inference
- **Flutter/Dart** - Mobile app integration

## Building

1. Reassemble models from parts:
```bash
# Models auto-reassemble during build
cargo build
```

2. The build process:
    - Combines model parts into complete models
    - Sets up Candle for memory-mapped loading
    - Prepares models for peer-to-peer coordination

## Development

Models are created from foundation models using:
1. Quantization (4-bit) for size reduction
2. Knowledge distillation for nano architectures
3. Compression and splitting for git-friendly storage

See `scripts/` for model creation and training tools.

## License

MPL-2.0