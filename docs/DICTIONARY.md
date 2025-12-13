# Dictionary & Corrections

## Overview

Two related systems:

1. **Dictionary** (`dictionary.rs`) - Fixes OCR errors in real-time
2. **Corrections** (`correction.rs`) - Stores user feedback for future training

## Dictionary: Fuzzy Text Correction

### Problem

OCR makes mistakes:
```
Raw OCR: "Uzer Authntication"
         Should be: "User Authentication"
```

### Solution

Match against known terms using Levenshtein distance:

```
"Uzer" → distance 1 from "User" → correct
"Authntication" → distance 2 from "Authentication" → correct
```

### Levenshtein Distance

Counts minimum edits (insert, delete, replace) to transform one string into another:

```
"kitten" → "sitting"

kitten
sitten  (replace k→s)
sittin  (replace e→i)
sitting (insert g)

Distance = 3
```

### Code

```rust
pub struct Dictionary {
    terms: HashMap<String, u32>,              // word → frequency
    corrections: HashMap<String, String>,     // wrong → right
    domains: HashMap<DomainSource, HashSet<String>>,
}

pub fn correct_phrase(&self, text: &str) -> CorrectionResult {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut corrected_words = Vec::new();

    for word in words {
        // Check explicit corrections first
        if let Some(right) = self.corrections.get(&word.to_lowercase()) {
            corrected_words.push(right.clone());
            continue;
        }

        // Find closest dictionary match
        let (best_match, distance) = self.find_closest(word);
        
        // Apply if close enough
        if self.should_correct(word, distance) {
            corrected_words.push(best_match);
        } else {
            corrected_words.push(word.to_string());
        }
    }

    CorrectionResult {
        original: text.to_string(),
        corrected: corrected_words.join(" "),
        confidence: /* based on distances */,
        was_corrected: /* any changes? */,
    }
}
```

### Distance Thresholds

Based on word length:

| Word Length | Max Distance |
|-------------|--------------|
| 0-3 chars | 1 |
| 4-6 chars | 2 |
| 7-12 chars | 3 |
| 13+ chars | 4 |

### Dictionary Sources

```rust
pub enum DomainSource {
    DiagramLabels,  // Current shapes on board
    CodeSymbols,    // From GitHub (class names, functions)
    Jira,           // Ticket titles
    Confluence,     // Page titles
    Personal,       // User's corrections
}
```

### Built-in Terms

```rust
impl DictionaryBuilder {
    pub fn with_common_terms(mut self) -> Self {
        // Flowchart
        self.add_terms(&["Start", "End", "Yes", "No", "Process", "Decision"]);
        
        // Architecture
        self.add_terms(&["User", "Client", "Server", "Database", "API", "Service"]);
        
        // Actions
        self.add_terms(&["Create", "Read", "Update", "Delete", "Login", "Logout"]);
        
        // Technical
        self.add_terms(&["HTTP", "REST", "JSON", "Token", "Cache", "Queue"]);
        
        self
    }
}
```

### Adding Context

The pipeline can add terms from the current context:

```rust
// Add terms from code symbols
pipeline.add_dictionary_terms(&["UserService", "AuthController", "OrderRepository"], DomainSource::CodeSymbols);

// Add user correction
pipeline.add_correction("Databse", "Database");
```

---

## Corrections: User Feedback Storage

### Purpose

When a user corrects model output, we store it for:
1. Immediate dictionary improvement
2. Future model retraining

### Schema

```sql
CREATE TABLE corrections (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    input_type TEXT NOT NULL,     -- "text", "image", "json"
    input_data TEXT NOT NULL,     -- The prompt or image hash
    original TEXT NOT NULL,       -- What model produced
    corrected TEXT NOT NULL,      -- What user fixed it to
    user_id TEXT NOT NULL,        -- XaeroID
    timestamp INTEGER NOT NULL,
    synced INTEGER DEFAULT 0,     -- Sent to peers?
    drained INTEGER DEFAULT 0     -- Used in training?
);
```

### Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    CORRECTION LIFECYCLE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. User corrects output                                        │
│     Model said: "flowchart TB\n  A --> B"                      │
│     User fixes: "flowchart TD\n  A[Start] --> B[End]"          │
│                                                                 │
│     synced = 0, drained = 0                                    │
│                                                                 │
│  2. Sync to group peers via Iroh                               │
│     Other team members see the correction                       │
│                                                                 │
│     synced = 1, drained = 0                                    │
│                                                                 │
│  3. XaeroFlux (cloud peer) collects corrections                │
│     Weekly batch for retraining                                │
│                                                                 │
│     synced = 1, drained = 1                                    │
│                                                                 │
│  4. Cleanup old drained corrections                            │
│     Keep storage bounded                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Code

```rust
// Log a correction
pub fn insert(db: &Connection, correction: &Correction) -> Result<()> {
    db.execute(
        "INSERT INTO corrections (...) VALUES (...)",
        params![
            correction.id,
            correction.model_id,
            correction.input_type,
            correction.input_data,
            correction.original,
            correction.corrected,
            correction.user_id,
            correction.timestamp,
            0,  // synced = false
            0,  // drained = false
        ],
    )?;
    Ok(())
}

// Get pending corrections for sync
pub fn list_pending(db: &Connection, limit: u32) -> Result<Vec<Correction>> {
    db.query("SELECT * FROM corrections WHERE synced = 0 LIMIT ?", [limit])
}

// Mark as synced
pub fn mark_synced(db: &Connection, id: &str) -> Result<()> {
    db.execute("UPDATE corrections SET synced = 1 WHERE id = ?", [id])
}

// Export for training (JSONL format)
pub fn export_for_training(db: &Connection, model_id: &str) -> Result<String> {
    let corrections = list_by_model(db, model_id)?;
    let mut output = String::new();

    for c in corrections.iter().filter(|c| !c.drained) {
        let example = json!({
            "messages": [
                { "role": "user", "content": c.input_data },
                { "role": "assistant", "content": c.corrected }
            ]
        });
        output.push_str(&serde_json::to_string(&example)?);
        output.push('\n');
    }

    Ok(output)
}
```

### P2P Sync Flow

```
┌─────────────────┐         ┌─────────────────┐
│  Cyan Node A    │         │  Cyan Node B    │
│                 │◄───────►│                 │
│  corrections:   │  Iroh   │  corrections:   │
│  - corr-001     │ gossip  │  - corr-001     │
│  - corr-002     │         │  - corr-002     │
└────────┬────────┘         └────────┬────────┘
         │                           │
         └──────────┬────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  XaeroFlux Cloud    │
         │  (just a peer)      │
         │                     │
         │  Collects all       │
         │  corrections        │
         │                     │
         │  Weekly: retrain    │
         │  Mark drained       │
         └─────────────────────┘
```

### Network Events

When a correction is logged, it broadcasts via cyan-backend:

```rust
pub enum AINetworkEvent {
    CorrectionLogged {
        correction_id: String,
        model_id: String,
        model_name: String,
        input_type: String,
        input_data: String,
        original: String,
        corrected: String,
        user_id: String,
        timestamp: i64,
    },
}
```

---

## Integration

### In Pipeline

```rust
// After OCR
let raw_text = trocr_output;
let corrected = self.dictionary.correct_phrase(&raw_text);
shape.text = Some(corrected.corrected);
```

### User Correction Flow

```rust
// User sees: "flowchart TB\n A --> B"
// User fixes to: "flowchart TD\n A[Start] --> B[End]"

// 1. Log correction
let correction = Correction {
    id: uuid(),
    model_id: "cyan-sketch",
    input_type: CorrectionInputType::Text,
    input_data: original_prompt,
    original: "flowchart TB\n A --> B",
    corrected: "flowchart TD\n A[Start] --> B[End]",
    user_id: current_user,
    timestamp: now(),
    synced: false,
    drained: false,
};
correction::insert(&db, &correction)?;

// 2. Also add to dictionary for immediate improvement
dictionary.add_correction("TB", "TD");  // If this was the pattern
```
