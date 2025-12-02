# Cyan Lens Training Plan

**Model**: cyan-lens (Phi-3-mini-4k-instruct + LoRA)
**Goal**: Design-aware AI that acts like Martin Fowler at your desk

---

## Current State

- **Base training**: 500 examples (design drift, over-architecture, stale designs)
- **Model**: design-analyst-q4.gguf (2.2GB)
- **Location**: huggingface.co/blockxaero/cyan-lens

---

## Training Sources

### Backend Patterns

| Source | URL | What to Extract | Examples |
|--------|-----|-----------------|----------|
| Refactoring Guru | refactoring.guru/design-patterns | 23 GoF patterns with code | 100 |
| Fowler Catalog | martinfowler.com/eaaCatalog | Enterprise patterns | 80 |
| Microservices.io | microservices.io/patterns | Distributed patterns | 60 |
| Azure Patterns | docs.microsoft.com/azure/architecture/patterns | Cloud patterns | 40 |

### Language-Specific

| Source | URL | Language | Examples |
|--------|-----|----------|----------|
| Rust Patterns | rust-unofficial.github.io/patterns | Rust | 60 |
| Effective Go | go.dev/doc/effective_go | Go | 50 |
| Go Patterns | github.com/tmrts/go-patterns | Go | 30 |
| Java Design Patterns | java-design-patterns.com | Java | 50 |
| Python Guide | docs.python-guide.org | Python | 40 |

### Frontend Patterns

| Source | URL | Platform | Examples |
|--------|-----|----------|----------|
| Swift by Sundell | swiftbysundell.com | Swift/iOS | 60 |
| Hacking with Swift | hackingwithswift.com | Swift/iOS | 40 |
| Flutter Docs | docs.flutter.dev | Flutter | 40 |
| BLoC Library | bloclibrary.dev | Flutter | 30 |
| Patterns.dev | patterns.dev | React/Web | 40 |
| React Docs | react.dev/learn | React | 30 |

### Concurrency

| Source | URL | Examples |
|--------|-----|----------|
| Tokio Tutorial | tokio.rs/tokio/tutorial | 20 |
| Go by Example | gobyexample.com | 30 |
| LMAX Disruptor | lmax-exchange.github.io/disruptor | 10 |

---

## Training Schedule

### Week 1 (TODAY - Day 1)

**Target**: Ship v2 model with core patterns

**Sources to scrape**:
- [ ] refactoring.guru - 23 GoF patterns
- [ ] martinfowler.com/eaaCatalog - 30 enterprise patterns
- [ ] microservices.io - 20 key patterns
- [ ] rust-unofficial.github.io/patterns - 20 idioms
- [ ] go.dev/doc/effective_go - 20 idioms

**Example generation**:
- [ ] 4 examples per pattern = ~400 new examples
- [ ] Combine with original 500 = 900 total

**Training**:
- [ ] LoRA on Phi-3-mini-4k-instruct
- [ ] 3 epochs
- [ ] Local (M-series) or HuggingFace

**Output**:
- [ ] cyan-lens-v2-q4.gguf
- [ ] Upload to HuggingFace

**Integration**:
- [ ] Rust: xaeroai crate loads model
- [ ] Swift: Chat UI calls Rust
- [ ] End-to-end test

---

### Week 2 (Future)

**Target**: Language-specific patterns

- [ ] Java patterns (java-design-patterns.com)
- [ ] Python idioms (docs.python-guide.org)
- [ ] Swift/iOS (swiftbysundell.com)
- [ ] Flutter (bloclibrary.dev)

---

### Week 3 (Future)

**Target**: Frontend + Web patterns

- [ ] React patterns (patterns.dev)
- [ ] Web architecture
- [ ] More Swift/Flutter depth

---

### Week 4 (Future)

**Target**: Concurrency + Advanced

- [ ] Tokio async patterns
- [ ] Go concurrency
- [ ] LMAX Disruptor
- [ ] Actor model patterns

---

## Today's Execution Plan

### Phase 1: Scrape (2 hours)

```
09:00 - 09:30  Setup scraping scripts
09:30 - 10:00  Scrape refactoring.guru (GoF)
10:00 - 10:30  Scrape martinfowler.com (Enterprise)
10:30 - 11:00  Scrape microservices.io + rust + go patterns
```

### Phase 2: Generate Examples (1.5 hours)

```
11:00 - 11:30  Generate GoF examples (Claude API)
11:30 - 12:00  Generate Enterprise examples
12:00 - 12:30  Generate language-specific examples
```

### Phase 3: Train (2 hours)

```
12:30 - 13:00  Combine datasets, validate
13:00 - 15:00  Train LoRA (local or HF)
```

### Phase 4: Merge + GGUF (1 hour)

```
15:00 - 15:30  Merge LoRA into Phi-3
15:30 - 16:00  Convert to GGUF, quantize Q4
```

### Phase 5: Swift Integration (2 hours)

```
16:00 - 17:00  Rust: xaeroai model loading + inference
17:00 - 18:00  Swift: Chat UI + FFI bridge
```

### Phase 6: Test (30 min)

```
18:00 - 18:30  End-to-end test
              - Test old capabilities (drift detection)
              - Test new capabilities (pattern analysis)
```

---

## File Structure

```
xaeroai/
├── data/
│   ├── design_analyst/
│   │   └── train.jsonl          # Original 500 examples
│   ├── patterns/
│   │   ├── gof.jsonl            # GoF patterns
│   │   ├── fowler.jsonl         # Enterprise patterns
│   │   ├── microservices.jsonl  # Distributed patterns
│   │   └── combined.jsonl       # All patterns merged
│   └── idioms/
│       ├── rust.jsonl
│       └── go.jsonl
├── scripts/
│   ├── scrape_patterns.py       # Scraping logic
│   ├── generate_examples.py     # Claude API generation
│   ├── combine_datasets.py      # Merge + shuffle
│   ├── train_lora.py            # Training script
│   └── convert_gguf.py          # GGUF conversion
├── models/
│   └── README.md                # Points to HuggingFace
├── src/
│   └── lib.rs                   # Rust inference code
└── Cargo.toml
```

---

## Quality Checklist

Before training:
- [ ] Each example has system/user/assistant messages
- [ ] User content > 50 chars
- [ ] Assistant content > 200 chars
- [ ] Assistant mentions pattern/issue/recommendation
- [ ] No duplicate examples
- [ ] Original 500 examples included (prevent forgetting)

After training:
- [ ] Test drift detection (old capability)
- [ ] Test over-architecture detection (old capability)
- [ ] Test GoF pattern identification (new)
- [ ] Test enterprise pattern recommendation (new)
- [ ] Test Rust idiom advice (new)
- [ ] Test Go idiom advice (new)

---

## Commands Reference

```bash
# Scrape
python scripts/scrape_patterns.py

# Generate examples
python scripts/generate_examples.py

# Combine datasets
python scripts/combine_datasets.py

# Train (local)
python scripts/train_lora.py \
  --base-model microsoft/Phi-3-mini-4k-instruct \
  --data data/patterns/combined.jsonl \
  --output checkpoints/cyan-lens-v2

# Merge LoRA
python scripts/merge_lora.py \
  --base microsoft/Phi-3-mini-4k-instruct \
  --lora checkpoints/cyan-lens-v2 \
  --output models/cyan-lens-v2-merged

# Convert to GGUF
python llama.cpp/convert_hf_to_gguf.py \
  models/cyan-lens-v2-merged \
  --outfile models/cyan-lens-v2-f16.gguf

# Quantize
./llama.cpp/llama-quantize \
  models/cyan-lens-v2-f16.gguf \
  models/cyan-lens-v2-q4.gguf q4_K_M

# Upload to HuggingFace
huggingface-cli upload blockxaero/cyan-lens \
  models/cyan-lens-v2-q4.gguf cyan-lens-q4.gguf
```

---

## Notes

- Always train on COMBINED dataset (old + new) to prevent forgetting
- Shuffle before training
- Train on base Phi-3, not on previously merged model
- Test both old and new capabilities after training