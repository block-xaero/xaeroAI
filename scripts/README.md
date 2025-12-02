# XaeroAI - Design Analyst Model

A fine-tuned Phi-3 model for software design analysis, project health assessment, and Mermaid diagram generation.

## Features

- **Project Health Analysis**: Analyze Jira/Slack integration data to identify design drift, stale tickets, and team bottlenecks
- **Design Pattern Knowledge**: GoF, enterprise, microservices, and Rust idiom patterns
- **Mermaid Diagram Generation**: Sequence diagrams, flowcharts, class diagrams (partial support for ER and state diagrams)

## Model

- **Base**: microsoft/Phi-3-mini-4k-instruct
- **Fine-tuning**: LoRA (rank 8, dropout 0.1)
- **Training**: MLX on Apple Silicon
- **Format**: GGUF (Q4_K_M quantized, ~2GB)

## Quick Start

### Using GGUF with llama.cpp
```bash
llama-cli -m models/design-analyst-v4-q4.gguf \
    -p "<|user|>Create a Mermaid sequenceDiagram for user login<|end|><|assistant|>" \
    -n 400
```

### Using Ollama
```bash
ollama create design-analyst -f Modelfile
ollama run design-analyst "Analyze this ticket: 45 days open, 67 Slack mentions, 0 PRs"
```

### Using MLX (Apple Silicon)
```bash
python -m mlx_lm generate \
    --model microsoft/Phi-3-mini-4k-instruct \
    --adapter-path adapters/design-analyst-v4 \
    --max-tokens 400 \
    --prompt "Create a Mermaid sequenceDiagram for OAuth2 flow"
```

## Project Structure

```
xaeroai/
├── adapters/                    # MLX LoRA adapters
│   └── design-analyst-v4/       # Current production adapter
├── models/                      # GGUF models (gitignored)
│   └── design-analyst-v4-q4.gguf
├── data/
│   └── mlx_final/              # Training data
│       ├── train.jsonl
│       └── valid.jsonl
├── scripts/
│   ├── train.py                # Train/continue training
│   ├── export.py               # Merge + GGUF conversion
│   ├── test.py                 # Test model outputs
│   └── generate_data.py        # Generate training data
├── lora_config.yaml            # Training configuration
├── Modelfile                   # Ollama model definition
└── README.md
```

## Scripts

### Train
```bash
# Train new version
python scripts/train.py --config lora_config.yaml

# Continue from checkpoint
python scripts/train.py --config lora_config.yaml --resume adapters/design-analyst-v4
```

### Export to GGUF
```bash
python scripts/export.py --adapter adapters/design-analyst-v4 --output models/design-analyst-v4
```

### Test
```bash
# Test with MLX
python scripts/test.py --use-mlx

# Test with GGUF
python scripts/test.py --model models/design-analyst-v4-q4.gguf
```

## Training Data

- **Project health examples**: ~1000 ticket analysis scenarios
- **Design patterns**: GoF, enterprise, Rust idioms
- **Mermaid diagrams**: ~250 examples (sequence, flowchart, class)

## Limitations

- classDiagram: ~33% valid syntax
- erDiagram: Not reliable
- stateDiagram: Not reliable

See `MODEL_IMPROVEMENT_CONTEXT.md` for improvement plan.

## License

BUSL - Business Source License. Copyright (c) Block Xaero Inc.