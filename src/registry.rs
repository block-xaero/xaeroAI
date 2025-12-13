//! Model registry - SQLite storage for model metadata
//!
//! Models are tied to boards (no orphan models).
//! Model files are synced via cyan-backend blob storage.
//! This registry stores metadata for discovery and loading.

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

/// Model record in registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    pub id: String,
    pub board_id: String,
    pub name: String,
    pub version: String,
    pub kind: String,              // gguf, onnx, lora
    pub capabilities: Vec<String>, // JSON array
    pub tags: Vec<String>,         // JSON array
    pub skill_md: String,          // Full SKILL.md content
    pub model_hash: String,        // Blake3 of model file
    pub file_id: Option<String>,   // Reference to synced file
    pub author: String,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Initialize the model_registry table
pub fn init_table(db: &Connection) -> Result<()> {
    db.execute(
        "CREATE TABLE IF NOT EXISTS model_registry (
            id TEXT PRIMARY KEY,
            board_id TEXT NOT NULL,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            kind TEXT NOT NULL,
            capabilities TEXT NOT NULL,
            tags TEXT,
            skill_md TEXT NOT NULL,
            model_hash TEXT NOT NULL,
            file_id TEXT,
            author TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )",
        [],
    )?;

    // Index for board queries
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_model_registry_board ON model_registry(board_id)",
        [],
    )?;

    // Index for capability queries
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_model_registry_kind ON model_registry(kind)",
        [],
    )?;

    Ok(())
}

/// Insert a new model record
pub fn insert(db: &Connection, record: &ModelRecord) -> Result<()> {
    let capabilities_json = serde_json::to_string(&record.capabilities)?;
    let tags_json = serde_json::to_string(&record.tags)?;

    db.execute(
        "INSERT INTO model_registry 
         (id, board_id, name, version, kind, capabilities, tags, skill_md, model_hash, file_id, author, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            record.id,
            record.board_id,
            record.name,
            record.version,
            record.kind,
            capabilities_json,
            tags_json,
            record.skill_md,
            record.model_hash,
            record.file_id,
            record.author,
            record.created_at,
            record.updated_at,
        ],
    )?;

    Ok(())
}

/// Get a model by ID
pub fn get(db: &Connection, id: &str) -> Result<Option<ModelRecord>> {
    let record = db.query_row(
        "SELECT id, board_id, name, version, kind, capabilities, tags, skill_md, model_hash, file_id, author, created_at, updated_at
         FROM model_registry WHERE id = ?1",
        params![id],
        |row| {
            let capabilities_json: String = row.get(5)?;
            let tags_json: String = row.get(6)?;
            
            Ok(ModelRecord {
                id: row.get(0)?,
                board_id: row.get(1)?,
                name: row.get(2)?,
                version: row.get(3)?,
                kind: row.get(4)?,
                capabilities: serde_json::from_str(&capabilities_json).unwrap_or_default(),
                tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                skill_md: row.get(7)?,
                model_hash: row.get(8)?,
                file_id: row.get(9)?,
                author: row.get(10)?,
                created_at: row.get(11)?,
                updated_at: row.get(12)?,
            })
        },
    ).optional()?;

    Ok(record)
}

/// List all models for a board
pub fn list_by_board(db: &Connection, board_id: &str) -> Result<Vec<ModelRecord>> {
    let mut stmt = db.prepare(
        "SELECT id, board_id, name, version, kind, capabilities, tags, skill_md, model_hash, file_id, author, created_at, updated_at
         FROM model_registry WHERE board_id = ?1 ORDER BY created_at DESC"
    )?;

    let records = stmt.query_map(params![board_id], |row| {
        let capabilities_json: String = row.get(5)?;
        let tags_json: String = row.get(6)?;
        
        Ok(ModelRecord {
            id: row.get(0)?,
            board_id: row.get(1)?,
            name: row.get(2)?,
            version: row.get(3)?,
            kind: row.get(4)?,
            capabilities: serde_json::from_str(&capabilities_json).unwrap_or_default(),
            tags: serde_json::from_str(&tags_json).unwrap_or_default(),
            skill_md: row.get(7)?,
            model_hash: row.get(8)?,
            file_id: row.get(9)?,
            author: row.get(10)?,
            created_at: row.get(11)?,
            updated_at: row.get(12)?,
        })
    })?.filter_map(|r| r.ok()).collect();

    Ok(records)
}

/// List all models with a specific capability
pub fn list_by_capability(db: &Connection, capability: &str) -> Result<Vec<ModelRecord>> {
    let mut stmt = db.prepare(
        "SELECT id, board_id, name, version, kind, capabilities, tags, skill_md, model_hash, file_id, author, created_at, updated_at
         FROM model_registry WHERE capabilities LIKE ?1 ORDER BY created_at DESC"
    )?;

    let pattern = format!("%\"{}%", capability);
    let records = stmt.query_map(params![pattern], |row| {
        let capabilities_json: String = row.get(5)?;
        let tags_json: String = row.get(6)?;
        
        Ok(ModelRecord {
            id: row.get(0)?,
            board_id: row.get(1)?,
            name: row.get(2)?,
            version: row.get(3)?,
            kind: row.get(4)?,
            capabilities: serde_json::from_str(&capabilities_json).unwrap_or_default(),
            tags: serde_json::from_str(&tags_json).unwrap_or_default(),
            skill_md: row.get(7)?,
            model_hash: row.get(8)?,
            file_id: row.get(9)?,
            author: row.get(10)?,
            created_at: row.get(11)?,
            updated_at: row.get(12)?,
        })
    })?.filter_map(|r| r.ok()).collect();

    Ok(records)
}

/// List all models of a specific kind
pub fn list_by_kind(db: &Connection, kind: &str) -> Result<Vec<ModelRecord>> {
    let mut stmt = db.prepare(
        "SELECT id, board_id, name, version, kind, capabilities, tags, skill_md, model_hash, file_id, author, created_at, updated_at
         FROM model_registry WHERE kind = ?1 ORDER BY created_at DESC"
    )?;

    let records = stmt.query_map(params![kind], |row| {
        let capabilities_json: String = row.get(5)?;
        let tags_json: String = row.get(6)?;
        
        Ok(ModelRecord {
            id: row.get(0)?,
            board_id: row.get(1)?,
            name: row.get(2)?,
            version: row.get(3)?,
            kind: row.get(4)?,
            capabilities: serde_json::from_str(&capabilities_json).unwrap_or_default(),
            tags: serde_json::from_str(&tags_json).unwrap_or_default(),
            skill_md: row.get(7)?,
            model_hash: row.get(8)?,
            file_id: row.get(9)?,
            author: row.get(10)?,
            created_at: row.get(11)?,
            updated_at: row.get(12)?,
        })
    })?.filter_map(|r| r.ok()).collect();

    Ok(records)
}

/// Update a model record
pub fn update(db: &Connection, record: &ModelRecord) -> Result<()> {
    let capabilities_json = serde_json::to_string(&record.capabilities)?;
    let tags_json = serde_json::to_string(&record.tags)?;

    db.execute(
        "UPDATE model_registry SET
         name = ?1, version = ?2, kind = ?3, capabilities = ?4, tags = ?5,
         skill_md = ?6, model_hash = ?7, file_id = ?8, author = ?9, updated_at = ?10
         WHERE id = ?11",
        params![
            record.name,
            record.version,
            record.kind,
            capabilities_json,
            tags_json,
            record.skill_md,
            record.model_hash,
            record.file_id,
            record.author,
            record.updated_at,
            record.id,
        ],
    )?;

    Ok(())
}

/// Delete a model by ID
pub fn delete(db: &Connection, id: &str) -> Result<()> {
    db.execute("DELETE FROM model_registry WHERE id = ?1", params![id])?;
    Ok(())
}

/// Delete all models for a board (housekeeping on board delete)
pub fn delete_by_board(db: &Connection, board_id: &str) -> Result<usize> {
    let count = db.execute(
        "DELETE FROM model_registry WHERE board_id = ?1",
        params![board_id],
    )?;
    Ok(count)
}

/// Search models by tag
pub fn search_by_tag(db: &Connection, tag: &str) -> Result<Vec<ModelRecord>> {
    let mut stmt = db.prepare(
        "SELECT id, board_id, name, version, kind, capabilities, tags, skill_md, model_hash, file_id, author, created_at, updated_at
         FROM model_registry WHERE tags LIKE ?1 ORDER BY created_at DESC"
    )?;

    let pattern = format!("%\"{}%", tag);
    let records = stmt.query_map(params![pattern], |row| {
        let capabilities_json: String = row.get(5)?;
        let tags_json: String = row.get(6)?;
        
        Ok(ModelRecord {
            id: row.get(0)?,
            board_id: row.get(1)?,
            name: row.get(2)?,
            version: row.get(3)?,
            kind: row.get(4)?,
            capabilities: serde_json::from_str(&capabilities_json).unwrap_or_default(),
            tags: serde_json::from_str(&tags_json).unwrap_or_default(),
            skill_md: row.get(7)?,
            model_hash: row.get(8)?,
            file_id: row.get(9)?,
            author: row.get(10)?,
            created_at: row.get(11)?,
            updated_at: row.get(12)?,
        })
    })?.filter_map(|r| r.ok()).collect();

    Ok(records)
}

/// Count models in registry
pub fn count(db: &Connection) -> Result<usize> {
    let count: i64 = db.query_row(
        "SELECT COUNT(*) FROM model_registry",
        [],
        |row| row.get(0),
    )?;
    Ok(count as usize)
}

/// Model registry wrapper for convenience
pub struct ModelRegistry;

impl ModelRegistry {
    pub fn init(db: &Connection) -> Result<()> {
        init_table(db)
    }

    pub fn insert(db: &Connection, record: &ModelRecord) -> Result<()> {
        insert(db, record)
    }

    pub fn get(db: &Connection, id: &str) -> Result<Option<ModelRecord>> {
        get(db, id)
    }

    pub fn list_by_board(db: &Connection, board_id: &str) -> Result<Vec<ModelRecord>> {
        list_by_board(db, board_id)
    }

    pub fn delete(db: &Connection, id: &str) -> Result<()> {
        delete(db, id)
    }

    pub fn delete_by_board(db: &Connection, board_id: &str) -> Result<usize> {
        delete_by_board(db, board_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_db() -> Connection {
        let db = Connection::open_in_memory().unwrap();
        init_table(&db).unwrap();
        db
    }

    #[test]
    fn test_insert_and_get() {
        let db = setup_db();
        
        let record = ModelRecord {
            id: "model-123".to_string(),
            board_id: "board-456".to_string(),
            name: "test-model".to_string(),
            version: "1.0.0".to_string(),
            kind: "gguf".to_string(),
            capabilities: vec!["text_generation".to_string()],
            tags: vec!["llm".to_string(), "phi".to_string()],
            skill_md: "# Test Model".to_string(),
            model_hash: "abc123".to_string(),
            file_id: Some("file-789".to_string()),
            author: "test".to_string(),
            created_at: 1234567890,
            updated_at: 1234567890,
        };

        insert(&db, &record).unwrap();
        
        let retrieved = get(&db, "model-123").unwrap().unwrap();
        assert_eq!(retrieved.name, "test-model");
        assert_eq!(retrieved.capabilities, vec!["text_generation"]);
    }

    #[test]
    fn test_list_by_board() {
        let db = setup_db();
        
        for i in 0..3 {
            let record = ModelRecord {
                id: format!("model-{}", i),
                board_id: "board-1".to_string(),
                name: format!("model-{}", i),
                version: "1.0.0".to_string(),
                kind: "gguf".to_string(),
                capabilities: vec![],
                tags: vec![],
                skill_md: "".to_string(),
                model_hash: "".to_string(),
                file_id: None,
                author: "".to_string(),
                created_at: i,
                updated_at: i,
            };
            insert(&db, &record).unwrap();
        }

        let models = list_by_board(&db, "board-1").unwrap();
        assert_eq!(models.len(), 3);
    }

    #[test]
    fn test_delete_by_board() {
        let db = setup_db();
        
        for i in 0..3 {
            let record = ModelRecord {
                id: format!("model-{}", i),
                board_id: "board-1".to_string(),
                name: format!("model-{}", i),
                version: "1.0.0".to_string(),
                kind: "gguf".to_string(),
                capabilities: vec![],
                tags: vec![],
                skill_md: "".to_string(),
                model_hash: "".to_string(),
                file_id: None,
                author: "".to_string(),
                created_at: i,
                updated_at: i,
            };
            insert(&db, &record).unwrap();
        }

        let deleted = delete_by_board(&db, "board-1").unwrap();
        assert_eq!(deleted, 3);
        
        let models = list_by_board(&db, "board-1").unwrap();
        assert_eq!(models.len(), 0);
    }
}
