---
name: cyan-lens
version: 0.4.0
kind: gguf
tags:
  - llm
  - mermaid
  - design-patterns
  - project-health
  - cyan-core
capabilities:
  - text_generation
  - text_to_mermaid
  - project_health
  - design_patterns
input:
  type: text
  formats: []
output:
  type: text
  formats: []
base_model: microsoft/Phi-3-mini-4k-instruct
lora_rank: 8
author: cyan
created: 1734048000
model_file: cyan-lens-q4.gguf
---

# Cyan Lens

Fine-tuned Phi-3 model for software design analysis and diagram generation.

## Overview

Cyan Lens is a design analyst assistant trained on:
- Software design patterns (GoF, enterprise, DDD, Rust idioms)
- Mermaid diagram generation
- Project health analysis from integration data

## Capabilities

### 1. Mermaid Diagram Generation

Generate valid Mermaid diagrams from natural language descriptions.

**Supported diagram types:**
- ✅ sequenceDiagram (100% valid syntax)
- ⚠️ flowchart (~66% valid)
- ⚠️ classDiagram (~33% valid)
- 🚧 erDiagram (in progress)
- 🚧 stateDiagram (in progress)

**Example:**
```
User: Create a sequence diagram for OAuth2 login flow
Assistant: 
sequenceDiagram
    participant User
    participant App
    participant AuthServer
    User->>App: Click Login
    App->>AuthServer: Redirect to /authorize
    AuthServer->>User: Show login form
    User->>AuthServer: Enter credentials
    AuthServer->>App: Authorization code
    App->>AuthServer: Exchange code for token
    AuthServer->>App: Access token
    App->>User: Logged in
```

### 2. Project Health Analysis

Analyze integration data (Slack, Jira, GitHub) to surface insights.

**Input format:**
```
<context>
Ticket: AUTH-101
Days open: 45
Slack mentions: 67
PRs: 0
Last update: 30 days ago
</context>
Analyze this ticket health
```

**Output:** JSON with risk assessment, recommendations, and severity.

### 3. Design Pattern Recommendations

Suggest appropriate design patterns based on problem description.

**Trained on:**
- Gang of Four patterns
- Enterprise patterns (Fowler)
- Domain-Driven Design
- Rust idioms and ownership patterns

## LoRA Adapters

The base model can be enhanced with specialized LoRA adapters:

| Adapter | Purpose | Status |
|---------|---------|--------|
| mermaid-flowchart | Flowchart generation | ✅ |
| mermaid-sequence | Sequence diagrams | ✅ |
| mermaid-class | Class diagrams | 🚧 |
| project-health | Integration analysis | ✅ |

## Usage

```rust
use xaeroai::{XaeroAI, InferenceRequest, InferenceInput};

let ai = XaeroAI::new("~/.cyan/models").await?;
ai.load_model("cyan-lens").await?;

let response = ai.infer(InferenceRequest {
    model: "cyan-lens".to_string(),
    input: InferenceInput::Text("Create a flowchart for error handling".to_string()),
    max_tokens: Some(512),
    temperature: Some(0.7),
}).await?;
```

## Model Details

- Base: Phi-3-mini-4k-instruct
- Quantization: Q4_K_M
- Context: 4096 tokens
- Size: ~2.3GB
- LoRA rank: 8

## Training

Trained using MLX on M3 Max:
- 600 iterations
- Batch size: 4
- Learning rate: 1e-5 with cosine decay
- Dropout: 0.1

See `/adapters/design-analyst-v4/` for weights.
