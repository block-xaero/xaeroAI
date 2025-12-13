//! Correction storage - SQLite for user feedback
//!
//! Corrections are logged locally, synced via Iroh P2P to XaeroFlux.
//! XaeroFlux collects corrections for periodic retraining.
//! Once incorporated into training, corrections are marked drained.

use anyhow::{anyhow, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Correction input type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CorrectionInputType {
    Text,
    Image,
    Json,
}

impl FromStr for CorrectionInputType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "image" => Ok(Self::Image),
            "json" => Ok(Self::Json),
            _ => Err(anyhow!("Unknown correction input type: {}", s)),
        }
    }
}

impl std::fmt::Display for CorrectionInputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Image => write!(f, "image"),
            Self::Json => write!(f, "json"),
        }
    }
}

/// A user correction to model output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correction {
    pub id: String,
    pub model_id: String,
    pub input_type: CorrectionInputType,
    pub input_data: String,        // Text prompt or Blake3 hash for images
    pub original: String,          // What model produced
    pub corrected: String,         // What user corrected it to
    pub user_id: String,           // XaeroID
    pub timestamp: i64,
    pub synced: bool,              // Synced to group peers
    pub drained: bool,             // Incorporated into training
}

/// Initialize the corrections table
pub fn init_table(db: &Connection) -> Result<()> {
    db.execute(
        "CREATE TABLE IF NOT EXISTS corrections (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            input_type TEXT NOT NULL,
            input_data TEXT NOT NULL,
            original TEXT NOT NULL,
            corrected TEXT NOT NULL,
            user_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            synced INTEGER DEFAULT 0,
            drained INTEGER DEFAULT 0
        )",
        [],
    )?;

    // Index for pending sync queries
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_corrections_synced ON corrections(synced)",
        [],
    )?;

    // Index for model queries
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_corrections_model ON corrections(model_id)",
        [],
    )?;

    Ok(())
}

/// Insert a new correction
pub fn insert(db: &Connection, correction: &Correction) -> Result<()> {
    db.execute(
        "INSERT INTO corrections 
         (id, model_id, input_type, input_data, original, corrected, user_id, timestamp, synced, drained)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            correction.id,
            correction.model_id,
            correction.input_type.to_string(),
            correction.input_data,
            correction.original,
            correction.corrected,
            correction.user_id,
            correction.timestamp,
            correction.synced as i32,
            correction.drained as i32,
        ],
    )?;

    Ok(())
}

/// Get a correction by ID
pub fn get(db: &Connection, id: &str) -> Result<Option<Correction>> {
    let correction = db.query_row(
        "SELECT id, model_id, input_type, input_data, original, corrected, user_id, timestamp, synced, drained
         FROM corrections WHERE id = ?1",
        params![id],
        |row| {
            let input_type_str: String = row.get(2)?;
            let synced: i32 = row.get(8)?;
            let drained: i32 = row.get(9)?;
            
            Ok(Correction {
                id: row.get(0)?,
                model_id: row.get(1)?,
                input_type: input_type_str.parse().unwrap_or(CorrectionInputType::Text),
                input_data: row.get(3)?,
                original: row.get(4)?,
                corrected: row.get(5)?,
                user_id: row.get(6)?,
                timestamp: row.get(7)?,
                synced: synced != 0,
                drained: drained != 0,
            })
        },
    ).optional()?;

    Ok(correction)
}

/// List pending (unsynced) corrections
pub fn list_pending(db: &Connection, limit: u32) -> Result<Vec<Correction>> {
    let mut stmt = db.prepare(
        "SELECT id, model_id, input_type, input_data, original, corrected, user_id, timestamp, synced, drained
         FROM corrections WHERE synced = 0 ORDER BY timestamp ASC LIMIT ?1"
    )?;

    let corrections = stmt.query_map(params![limit], |row| {
        let input_type_str: String = row.get(2)?;
        let synced: i32 = row.get(8)?;
        let drained: i32 = row.get(9)?;
        
        Ok(Correction {
            id: row.get(0)?,
            model_id: row.get(1)?,
            input_type: input_type_str.parse().unwrap_or(CorrectionInputType::Text),
            input_data: row.get(3)?,
            original: row.get(4)?,
            corrected: row.get(5)?,
            user_id: row.get(6)?,
            timestamp: row.get(7)?,
            synced: synced != 0,
            drained: drained != 0,
        })
    })?.filter_map(|r| r.ok()).collect();

    Ok(corrections)
}

/// List corrections for a model
pub fn list_by_model(db: &Connection, model_id: &str) -> Result<Vec<Correction>> {
    let mut stmt = db.prepare(
        "SELECT id, model_id, input_type, input_data, original, corrected, user_id, timestamp, synced, drained
         FROM corrections WHERE model_id = ?1 ORDER BY timestamp DESC"
    )?;

    let corrections = stmt.query_map(params![model_id], |row| {
        let input_type_str: String = row.get(2)?;
        let synced: i32 = row.get(8)?;
        let drained: i32 = row.get(9)?;
        
        Ok(Correction {
            id: row.get(0)?,
            model_id: row.get(1)?,
            input_type: input_type_str.parse().unwrap_or(CorrectionInputType::Text),
            input_data: row.get(3)?,
            original: row.get(4)?,
            corrected: row.get(5)?,
            user_id: row.get(6)?,
            timestamp: row.get(7)?,
            synced: synced != 0,
            drained: drained != 0,
        })
    })?.filter_map(|r| r.ok()).collect();

    Ok(corrections)
}

/// List undrained corrections (for training export)
pub fn list_undrained(db: &Connection, limit: u32) -> Result<Vec<Correction>> {
    let mut stmt = db.prepare(
        "SELECT id, model_id, input_type, input_data, original, corrected, user_id, timestamp, synced, drained
         FROM corrections WHERE drained = 0 AND synced = 1 ORDER BY timestamp ASC LIMIT ?1"
    )?;

    let corrections = stmt.query_map(params![limit], |row| {
        let input_type_str: String = row.get(2)?;
        let synced: i32 = row.get(8)?;
        let drained: i32 = row.get(9)?;
        
        Ok(Correction {
            id: row.get(0)?,
            model_id: row.get(1)?,
            input_type: input_type_str.parse().unwrap_or(CorrectionInputType::Text),
            input_data: row.get(3)?,
            original: row.get(4)?,
            corrected: row.get(5)?,
            user_id: row.get(6)?,
            timestamp: row.get(7)?,
            synced: synced != 0,
            drained: drained != 0,
        })
    })?.filter_map(|r| r.ok()).collect();

    Ok(corrections)
}

/// Mark a correction as synced
pub fn mark_synced(db: &Connection, id: &str) -> Result<()> {
    db.execute(
        "UPDATE corrections SET synced = 1 WHERE id = ?1",
        params![id],
    )?;
    Ok(())
}

/// Mark a correction as drained (incorporated into training)
pub fn mark_drained(db: &Connection, id: &str) -> Result<()> {
    db.execute(
        "UPDATE corrections SET drained = 1 WHERE id = ?1",
        params![id],
    )?;
    Ok(())
}

/// Delete a correction
pub fn delete(db: &Connection, id: &str) -> Result<()> {
    db.execute("DELETE FROM corrections WHERE id = ?1", params![id])?;
    Ok(())
}

/// Delete all corrections for a model (housekeeping)
pub fn delete_by_model(db: &Connection, model_id: &str) -> Result<usize> {
    let count = db.execute(
        "DELETE FROM corrections WHERE model_id = ?1",
        params![model_id],
    )?;
    Ok(count)
}

/// Delete drained corrections older than timestamp (cleanup)
pub fn cleanup_drained(db: &Connection, before_timestamp: i64) -> Result<usize> {
    let count = db.execute(
        "DELETE FROM corrections WHERE drained = 1 AND timestamp < ?1",
        params![before_timestamp],
    )?;
    Ok(count)
}

/// Export corrections for training (JSONL format)
pub fn export_for_training(db: &Connection, model_id: &str) -> Result<String> {
    let corrections = list_by_model(db, model_id)?;
    let mut output = String::new();

    for c in corrections.iter().filter(|c| !c.drained) {
        let training_example = serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": c.input_data
                },
                {
                    "role": "assistant",
                    "content": c.corrected
                }
            ]
        });

        output.push_str(&serde_json::to_string(&training_example)?);
        output.push('\n');
    }

    Ok(output)
}

/// Count pending corrections
pub fn count_pending(db: &Connection) -> Result<usize> {
    let count: i64 = db.query_row(
        "SELECT COUNT(*) FROM corrections WHERE synced = 0",
        [],
        |row| row.get(0),
    )?;
    Ok(count as usize)
}

/// Count total corrections
pub fn count(db: &Connection) -> Result<usize> {
    let count: i64 = db.query_row(
        "SELECT COUNT(*) FROM corrections",
        [],
        |row| row.get(0),
    )?;
    Ok(count as usize)
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
        
        let correction = Correction {
            id: "corr-123".to_string(),
            model_id: "model-456".to_string(),
            input_type: CorrectionInputType::Text,
            input_data: "Generate a flowchart".to_string(),
            original: "flowchart TB\n  A --> B".to_string(),
            corrected: "flowchart TD\n  A[Start] --> B[End]".to_string(),
            user_id: "user-789".to_string(),
            timestamp: 1234567890,
            synced: false,
            drained: false,
        };

        insert(&db, &correction).unwrap();
        
        let retrieved = get(&db, "corr-123").unwrap().unwrap();
        assert_eq!(retrieved.model_id, "model-456");
        assert!(!retrieved.synced);
        assert!(!retrieved.drained);
    }

    #[test]
    fn test_list_pending() {
        let db = setup_db();
        
        for i in 0..5 {
            let correction = Correction {
                id: format!("corr-{}", i),
                model_id: "model-1".to_string(),
                input_type: CorrectionInputType::Text,
                input_data: format!("input-{}", i),
                original: "original".to_string(),
                corrected: "corrected".to_string(),
                user_id: "user".to_string(),
                timestamp: i,
                synced: i % 2 == 0, // Even ones are synced
                drained: false,
            };
            insert(&db, &correction).unwrap();
        }

        let pending = list_pending(&db, 10).unwrap();
        assert_eq!(pending.len(), 2); // Only unsynced ones
    }

    #[test]
    fn test_mark_synced() {
        let db = setup_db();
        
        let correction = Correction {
            id: "corr-1".to_string(),
            model_id: "model-1".to_string(),
            input_type: CorrectionInputType::Text,
            input_data: "input".to_string(),
            original: "original".to_string(),
            corrected: "corrected".to_string(),
            user_id: "user".to_string(),
            timestamp: 0,
            synced: false,
            drained: false,
        };
        insert(&db, &correction).unwrap();

        mark_synced(&db, "corr-1").unwrap();
        
        let retrieved = get(&db, "corr-1").unwrap().unwrap();
        assert!(retrieved.synced);
    }

    #[test]
    fn test_export_for_training() {
        let db = setup_db();
        
        let correction = Correction {
            id: "corr-1".to_string(),
            model_id: "model-1".to_string(),
            input_type: CorrectionInputType::Text,
            input_data: "Create a flowchart".to_string(),
            original: "bad output".to_string(),
            corrected: "flowchart TD\n  A --> B".to_string(),
            user_id: "user".to_string(),
            timestamp: 0,
            synced: true,
            drained: false,
        };
        insert(&db, &correction).unwrap();

        let export = export_for_training(&db, "model-1").unwrap();
        assert!(export.contains("Create a flowchart"));
        assert!(export.contains("flowchart TD"));
    }
}
