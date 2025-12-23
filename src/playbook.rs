//! ACE-style Playbook with SQLite + FTS5 keyword search
//!
//! Playbook bullets are itemized knowledge with quality counters.
//! Based on the ACE paper (arxiv.org/pdf/2510.04618):
//! - Bullets have unique IDs, sections, and content
//! - Quality tracked via helpful/harmful/neutral counters
//! - FTS5 for fast keyword-based retrieval

use anyhow::{anyhow, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Playbook section types (ACE-style)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Section {
    /// High-level strategies and hard rules
    Strategies,
    /// API-specific patterns and schemas
    Apis,
    /// Common mistakes to avoid
    Mistakes,
    /// Formulas and calculations
    Formulas,
    /// Verification checklists
    Verification,
}

impl Section {
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::Strategies => "str",
            Self::Apis => "api",
            Self::Mistakes => "mis",
            Self::Formulas => "for",
            Self::Verification => "ver",
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Strategies => "strategies",
            Self::Apis => "apis",
            Self::Mistakes => "mistakes",
            Self::Formulas => "formulas",
            Self::Verification => "verification",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "strategies" => Self::Strategies,
            "apis" => Self::Apis,
            "mistakes" => Self::Mistakes,
            "formulas" => Self::Formulas,
            "verification" => Self::Verification,
            _ => Self::Strategies,
        }
    }

    pub fn header(&self) -> &'static str {
        match self {
            Self::Strategies => "## Strategies and Hard Rules",
            Self::Apis => "## APIs and Schemas",
            Self::Mistakes => "## Common Mistakes to Avoid",
            Self::Formulas => "## Formulas and Calculations",
            Self::Verification => "## Verification Checklist",
        }
    }
}

/// Feedback tag from generator (ACE-style)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeedbackTag {
    Helpful,
    Harmful,
    Neutral,
}

impl FeedbackTag {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Helpful => "helpful",
            Self::Harmful => "harmful",
            Self::Neutral => "neutral",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "helpful" => Self::Helpful,
            "harmful" => Self::Harmful,
            _ => Self::Neutral,
        }
    }
}

/// A single playbook bullet (ACE-style)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bullet {
    pub id: String,
    pub scope: String,
    pub section: Section,
    pub content: String,

    // Quality counters
    pub helpful_count: u32,
    pub harmful_count: u32,
    pub neutral_count: u32,

    // Computed score (helpful / (helpful + harmful))
    pub score: f32,

    // Source tracking
    pub source_type: Option<String>,
    pub source_id: Option<String>,

    // Timestamps
    pub created_at: i64,
    pub updated_at: i64,
}

impl Bullet {
    /// Format bullet for prompt injection (ACE format)
    pub fn format(&self) -> String {
        format!(
            "[{}] helpful={} harmful={} :: {}",
            self.id, self.helpful_count, self.harmful_count, self.content
        )
    }
}

/// Initialize playbook tables in SQLite
pub fn init_tables(db: &Connection) -> Result<()> {
    db.execute_batch(r#"
        -- Main bullets table
        CREATE TABLE IF NOT EXISTS playbook_bullets (
            id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            section TEXT NOT NULL,
            content TEXT NOT NULL,
            helpful_count INTEGER DEFAULT 0,
            harmful_count INTEGER DEFAULT 0,
            neutral_count INTEGER DEFAULT 0,
            score REAL GENERATED ALWAYS AS (
                CASE WHEN (helpful_count + harmful_count) = 0 THEN 0.5
                     ELSE CAST(helpful_count AS REAL) / (helpful_count + harmful_count)
                END
            ) STORED,
            source_type TEXT,
            source_id TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );

        -- Indexes for fast lookup
        CREATE INDEX IF NOT EXISTS idx_playbook_scope ON playbook_bullets(scope);
        CREATE INDEX IF NOT EXISTS idx_playbook_section ON playbook_bullets(scope, section);
        CREATE INDEX IF NOT EXISTS idx_playbook_score ON playbook_bullets(scope, score DESC);

        -- ID sequence table
        CREATE TABLE IF NOT EXISTS playbook_sequences (
            prefix TEXT PRIMARY KEY,
            next_id INTEGER DEFAULT 1
        );

        -- FTS5 virtual table for keyword search
        CREATE VIRTUAL TABLE IF NOT EXISTS playbook_fts USING fts5(
            id,
            content,
            tokenize='porter unicode61'
        );
    "#)?;

    // Create triggers for FTS sync (ignore errors if already exist)
    let _ = db.execute_batch(r#"
        CREATE TRIGGER playbook_ai AFTER INSERT ON playbook_bullets BEGIN
            INSERT INTO playbook_fts(id, content) VALUES (new.id, new.content);
        END;

        CREATE TRIGGER playbook_ad AFTER DELETE ON playbook_bullets BEGIN
            DELETE FROM playbook_fts WHERE id = old.id;
        END;

        CREATE TRIGGER playbook_au AFTER UPDATE OF content ON playbook_bullets BEGIN
            DELETE FROM playbook_fts WHERE id = old.id;
            INSERT INTO playbook_fts(id, content) VALUES (new.id, new.content);
        END;
    "#);

    Ok(())
}

/// Generate next bullet ID for scope+section
pub fn next_id(db: &Connection, scope: &str, section: Section) -> Result<String> {
    let prefix = format!("{}:{}", scope, section.prefix());

    db.execute(
        "INSERT INTO playbook_sequences (prefix, next_id) VALUES (?1, 1)
         ON CONFLICT(prefix) DO UPDATE SET next_id = next_id + 1",
        params![prefix],
    )?;

    let next: i64 = db.query_row(
        "SELECT next_id FROM playbook_sequences WHERE prefix = ?1",
        params![prefix],
        |row| row.get(0),
    )?;

    Ok(format!("{}-{:05}", section.prefix(), next - 1))
}

/// Add new bullet
pub fn add(db: &Connection, scope: &str, section: Section, content: &str) -> Result<String> {
    let id = next_id(db, scope, section)?;
    let now = chrono::Utc::now().timestamp();

    db.execute(
        "INSERT INTO playbook_bullets
         (id, scope, section, content, helpful_count, harmful_count, neutral_count, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, 0, 0, 0, ?5, ?5)",
        params![id, scope, section.as_str(), content, now],
    )?;

    Ok(id)
}

/// Add bullet with source tracking
pub fn add_with_source(
    db: &Connection,
    scope: &str,
    section: Section,
    content: &str,
    source_type: &str,
    source_id: &str,
) -> Result<String> {
    let id = next_id(db, scope, section)?;
    let now = chrono::Utc::now().timestamp();

    db.execute(
        "INSERT INTO playbook_bullets
         (id, scope, section, content, helpful_count, harmful_count, neutral_count,
          source_type, source_id, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, 0, 0, 0, ?5, ?6, ?7, ?7)",
        params![id, scope, section.as_str(), content, source_type, source_id, now],
    )?;

    Ok(id)
}

/// Get bullet by ID
pub fn get(db: &Connection, id: &str) -> Result<Option<Bullet>> {
    let result = db.query_row(
        "SELECT id, scope, section, content, helpful_count, harmful_count, neutral_count,
                score, source_type, source_id, created_at, updated_at
         FROM playbook_bullets WHERE id = ?1",
        params![id],
        row_to_bullet,
    ).optional()?;

    Ok(result)
}

/// Record feedback on a bullet (ACE: update counters)
pub fn record_feedback(db: &Connection, bullet_id: &str, tag: FeedbackTag) -> Result<()> {
    let column = match tag {
        FeedbackTag::Helpful => "helpful_count",
        FeedbackTag::Harmful => "harmful_count",
        FeedbackTag::Neutral => "neutral_count",
    };

    let now = chrono::Utc::now().timestamp();
    let updated = db.execute(
        &format!(
            "UPDATE playbook_bullets SET {} = {} + 1, updated_at = ?1 WHERE id = ?2",
            column, column
        ),
        params![now, bullet_id],
    )?;

    if updated == 0 {
        return Err(anyhow!("Bullet not found: {}", bullet_id));
    }

    Ok(())
}

/// Retrieve bullets by FTS5 keyword search
pub fn retrieve(db: &Connection, scope: &str, query: &str, limit: usize) -> Result<Vec<Bullet>> {
    // Extract keywords (words > 2 chars)
    let keywords: Vec<&str> = query
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .collect();

    if keywords.is_empty() {
        // No keywords, return top bullets by score
        return retrieve_top(db, scope, limit);
    }

    // Build FTS query (OR for any match)
    let fts_query = keywords.join(" OR ");

    let mut stmt = db.prepare(
        "SELECT b.id, b.scope, b.section, b.content,
                b.helpful_count, b.harmful_count, b.neutral_count, b.score,
                b.source_type, b.source_id, b.created_at, b.updated_at
         FROM playbook_bullets b
         JOIN playbook_fts fts ON b.id = fts.id
         WHERE b.scope = ?1 AND playbook_fts MATCH ?2
         ORDER BY b.score DESC
         LIMIT ?3"
    )?;

    let bullets = stmt.query_map(params![scope, fts_query, limit as i64], row_to_bullet)?
        .filter_map(|r| r.ok())
        .collect();

    Ok(bullets)
}

/// Retrieve top bullets by score (no keyword filter)
pub fn retrieve_top(db: &Connection, scope: &str, limit: usize) -> Result<Vec<Bullet>> {
    let mut stmt = db.prepare(
        "SELECT id, scope, section, content, helpful_count, harmful_count, neutral_count,
                score, source_type, source_id, created_at, updated_at
         FROM playbook_bullets
         WHERE scope = ?1
         ORDER BY score DESC
         LIMIT ?2"
    )?;

    let bullets = stmt.query_map(params![scope, limit as i64], row_to_bullet)?
        .filter_map(|r| r.ok())
        .collect();

    Ok(bullets)
}

/// Retrieve bullets by section
pub fn retrieve_by_section(db: &Connection, scope: &str, section: Section, limit: usize) -> Result<Vec<Bullet>> {
    let mut stmt = db.prepare(
        "SELECT id, scope, section, content, helpful_count, harmful_count, neutral_count,
                score, source_type, source_id, created_at, updated_at
         FROM playbook_bullets
         WHERE scope = ?1 AND section = ?2
         ORDER BY score DESC
         LIMIT ?3"
    )?;

    let bullets = stmt.query_map(params![scope, section.as_str(), limit as i64], row_to_bullet)?
        .filter_map(|r| r.ok())
        .collect();

    Ok(bullets)
}

/// Build playbook context for prompt injection (ACE format)
pub fn build_context(db: &Connection, scope: &str, query: Option<&str>, max_bullets: usize) -> Result<String> {
    let mut context = String::from("PLAYBOOK_BEGIN\n");

    // If query provided, do keyword retrieval
    let relevant_bullets = if let Some(q) = query {
        retrieve(db, scope, q, max_bullets / 2)?
    } else {
        vec![]
    };

    // Add relevant bullets first
    if !relevant_bullets.is_empty() {
        context.push_str("\n## Relevant to Current Query\n");
        for bullet in &relevant_bullets {
            context.push_str(&bullet.format());
            context.push('\n');
        }
    }

    // Add top bullets by section
    let sections = [
        Section::Strategies,
        Section::Apis,
        Section::Mistakes,
        Section::Formulas,
        Section::Verification,
    ];

    let remaining = max_bullets.saturating_sub(relevant_bullets.len());
    let per_section = remaining / sections.len();

    for section in sections {
        let bullets = retrieve_by_section(db, scope, section, per_section)?;

        // Skip bullets already included
        let new_bullets: Vec<_> = bullets.iter()
            .filter(|b| !relevant_bullets.iter().any(|r| r.id == b.id))
            .collect();

        if !new_bullets.is_empty() {
            context.push_str(&format!("\n{}\n", section.header()));
            for bullet in new_bullets {
                context.push_str(&bullet.format());
                context.push('\n');
            }
        }
    }

    context.push_str("\nPLAYBOOK_END\n");
    Ok(context)
}

/// Delete a bullet
pub fn delete(db: &Connection, bullet_id: &str) -> Result<()> {
    db.execute(
        "DELETE FROM playbook_bullets WHERE id = ?1",
        params![bullet_id],
    )?;

    Ok(())
}

/// Count bullets in scope
pub fn count(db: &Connection, scope: &str) -> Result<usize> {
    let count: i64 = db.query_row(
        "SELECT COUNT(*) FROM playbook_bullets WHERE scope = ?1",
        params![scope],
        |row| row.get(0),
    )?;
    Ok(count as usize)
}

/// Get playbook stats
pub fn stats(db: &Connection, scope: &str) -> Result<PlaybookStats> {
    let total = count(db, scope)?;

    let mut by_section: HashMap<String, usize> = HashMap::new();
    let mut stmt = db.prepare(
        "SELECT section, COUNT(*) FROM playbook_bullets WHERE scope = ?1 GROUP BY section"
    )?;

    let rows = stmt.query_map(params![scope], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;

    for row in rows {
        let (section, cnt) = row?;
        by_section.insert(section, cnt as usize);
    }

    let avg_score: f64 = db.query_row(
        "SELECT COALESCE(AVG(score), 0.5) FROM playbook_bullets WHERE scope = ?1",
        params![scope],
        |row| row.get(0),
    )?;

    Ok(PlaybookStats {
        total_bullets: total,
        by_section,
        avg_score,
    })
}

/// List all bullets in scope (for export/debug)
pub fn list_all(db: &Connection, scope: &str) -> Result<Vec<Bullet>> {
    let mut stmt = db.prepare(
        "SELECT id, scope, section, content, helpful_count, harmful_count, neutral_count,
                score, source_type, source_id, created_at, updated_at
         FROM playbook_bullets
         WHERE scope = ?1
         ORDER BY section, score DESC"
    )?;

    let bullets = stmt.query_map(params![scope], row_to_bullet)?
        .filter_map(|r| r.ok())
        .collect();

    Ok(bullets)
}

/// List active bullets (score >= 0, used for prompt injection)
pub fn list_active(db: &Connection, scope: &str) -> Result<Vec<Bullet>> {
    let mut stmt = db.prepare(
        "SELECT id, scope, section, content, helpful_count, harmful_count, neutral_count,
                score, source_type, source_id, created_at, updated_at
         FROM playbook_bullets
         WHERE scope = ?1 AND score >= 0
         ORDER BY score DESC
         LIMIT 20"
    )?;

    let bullets = stmt.query_map(params![scope], row_to_bullet)?
        .filter_map(|r| r.ok())
        .collect();

    Ok(bullets)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybookStats {
    pub total_bullets: usize,
    pub by_section: HashMap<String, usize>,
    pub avg_score: f64,
}

fn row_to_bullet(row: &rusqlite::Row) -> rusqlite::Result<Bullet> {
    let section_str: String = row.get(2)?;

    Ok(Bullet {
        id: row.get(0)?,
        scope: row.get(1)?,
        section: Section::from_str(&section_str),
        content: row.get(3)?,
        helpful_count: row.get(4)?,
        harmful_count: row.get(5)?,
        neutral_count: row.get(6)?,
        score: row.get(7)?,
        source_type: row.get(8)?,
        source_id: row.get(9)?,
        created_at: row.get(10)?,
        updated_at: row.get(11)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_db() -> Connection {
        let db = Connection::open_in_memory().unwrap();
        init_tables(&db).unwrap();
        db
    }

    #[test]
    fn test_add_and_get() {
        let db = setup_db();

        let id = add(&db, "test-scope", Section::Strategies, "Test bullet content").unwrap();
        assert!(id.starts_with("str-"));

        let bullet = get(&db, &id).unwrap().unwrap();
        assert_eq!(bullet.content, "Test bullet content");
        assert_eq!(bullet.helpful_count, 0);
        assert_eq!(bullet.score, 0.5); // Default when no feedback
    }

    #[test]
    fn test_feedback() {
        let db = setup_db();

        let id = add(&db, "test", Section::Strategies, "Test").unwrap();

        record_feedback(&db, &id, FeedbackTag::Helpful).unwrap();
        record_feedback(&db, &id, FeedbackTag::Helpful).unwrap();
        record_feedback(&db, &id, FeedbackTag::Harmful).unwrap();

        let bullet = get(&db, &id).unwrap().unwrap();
        assert_eq!(bullet.helpful_count, 2);
        assert_eq!(bullet.harmful_count, 1);
        // score = 2 / (2 + 1) = 0.666...
        assert!((bullet.score - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_fts_retrieval() {
        let db = setup_db();

        add(&db, "test", Section::Strategies, "Search by group name not board name").unwrap();
        add(&db, "test", Section::Mistakes, "Don't use DELETE without WHERE clause").unwrap();
        add(&db, "test", Section::Apis, "groups table has id, name, icon columns").unwrap();

        // Search for "group"
        let results = retrieve(&db, "test", "group name", 10).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|b| b.content.contains("group")));
    }

    #[test]
    fn test_build_context() {
        let db = setup_db();

        add(&db, "test", Section::Strategies, "Strategy 1").unwrap();
        add(&db, "test", Section::Mistakes, "Mistake 1").unwrap();

        let context = build_context(&db, "test", Some("strategy"), 10).unwrap();

        assert!(context.contains("PLAYBOOK_BEGIN"));
        assert!(context.contains("PLAYBOOK_END"));
        assert!(context.contains("Strategy 1"));
    }

    #[test]
    fn test_stats() {
        let db = setup_db();

        add(&db, "test", Section::Strategies, "S1").unwrap();
        add(&db, "test", Section::Strategies, "S2").unwrap();
        add(&db, "test", Section::Mistakes, "M1").unwrap();

        let stats = stats(&db, "test").unwrap();
        assert_eq!(stats.total_bullets, 3);
        assert_eq!(stats.by_section.get("strategies"), Some(&2));
        assert_eq!(stats.by_section.get("mistakes"), Some(&1));
    }

    #[test]
    fn test_bullet_format() {
        let bullet = Bullet {
            id: "str-00001".to_string(),
            scope: "test".to_string(),
            section: Section::Strategies,
            content: "Test content".to_string(),
            helpful_count: 5,
            harmful_count: 2,
            neutral_count: 1,
            score: 0.714,
            source_type: None,
            source_id: None,
            created_at: 0,
            updated_at: 0,
        };

        let formatted = bullet.format();
        assert_eq!(formatted, "[str-00001] helpful=5 harmful=2 :: Test content");
    }
}