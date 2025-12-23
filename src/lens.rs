//! CyanLens - AI-powered workspace search
//!
//! Flow:
//! 1. User query → retrieve relevant playbook bullets (FTS5)
//! 2. Build prompt with playbook context + tool definitions
//! 3. Phi generates SQL
//! 4. Execute SQL against cyan.db
//! 5. Format results with deep links
//! 6. User feedback → update playbook

use crate::playbook::{self, Bullet, FeedbackTag, Section};
use crate::runtime::{InferenceInput, InferenceOutput, Runtime};
use crate::skill::Skill;
use anyhow::{anyhow, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

/// Search result with deep link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub name: String,
    pub result_type: String,  // "board", "workspace", "group", "cell"
    pub snippet: Option<String>,
    pub deep_link: String,
    pub metadata: Option<serde_json::Value>,
}

/// Lens search response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensResponse {
    pub request_id: String,
    pub query: String,
    pub generated_sql: Option<String>,
    pub results: Vec<SearchResult>,
    pub playbook_bullets_used: Vec<String>,  // IDs of bullets used in prompt
    pub latency_ms: u64,
    pub error: Option<String>,
}

/// Feedback for a lens query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensFeedback {
    pub request_id: String,
    pub was_helpful: bool,
    pub bullet_feedback: Vec<BulletFeedback>,
    pub correction: Option<LensCorrection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulletFeedback {
    pub bullet_id: String,
    pub tag: String,  // "helpful", "harmful", "neutral"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensCorrection {
    pub wrong_sql: String,
    pub correct_sql: Option<String>,
    pub explanation: String,  // What the user actually wanted
}

/// CyanLens engine
pub struct CyanLens {
    scope: String,
    schema_context: String,
}

impl CyanLens {
    /// Create new CyanLens with schema context
    pub fn new(scope: &str) -> Self {
        Self {
            scope: scope.to_string(),
            schema_context: Self::default_schema_context(),
        }
    }

    /// Default schema context for Cyan workspace
    fn default_schema_context() -> String {
        r#"
Available SQLite tables:

## groups
- id TEXT PRIMARY KEY
- name TEXT NOT NULL
- icon TEXT
- color TEXT
- created_at INTEGER

## workspaces
- id TEXT PRIMARY KEY
- group_id TEXT REFERENCES groups(id)
- name TEXT NOT NULL
- created_at INTEGER

## objects (boards)
- id TEXT PRIMARY KEY
- workspace_id TEXT REFERENCES workspaces(id)
- name TEXT NOT NULL
- type TEXT DEFAULT 'Whiteboard'
- created_at INTEGER
- last_accessed INTEGER

## notebook_cells
- id TEXT PRIMARY KEY
- board_id TEXT REFERENCES objects(id)
- cell_type TEXT NOT NULL  -- 'markdown', 'mermaid', 'code', 'image'
- content TEXT
- output TEXT
- position INTEGER

## board_metadata
- board_id TEXT PRIMARY KEY REFERENCES objects(id)
- labels TEXT  -- JSON array of strings
- rating INTEGER  -- 1-5 stars
- contains_model INTEGER  -- 0 or 1
- contains_skill INTEGER  -- 0 or 1

Common queries:
- Find boards by name: SELECT * FROM objects WHERE name LIKE '%keyword%'
- Find boards by group: SELECT o.* FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'GroupName'
- Find boards with mermaid: SELECT DISTINCT o.* FROM objects o JOIN notebook_cells c ON c.board_id = o.id WHERE c.cell_type = 'mermaid'
- Recent boards: SELECT * FROM objects ORDER BY last_accessed DESC LIMIT 10
"#.to_string()
    }

    /// Search using natural language query
    pub fn search(
        &self,
        runtime: &mut Runtime,
        model_id: &str,
        playbook_db: &Connection,
        cyan_db: &Connection,
        request_id: &str,
        query: &str,
    ) -> Result<LensResponse> {
        let start = std::time::Instant::now();

        // 1. Retrieve relevant playbook bullets
        let bullets = playbook::retrieve(playbook_db, &self.scope, query, 5)
            .unwrap_or_default();
        let bullet_ids: Vec<String> = bullets.iter().map(|b| b.id.clone()).collect();

        // 2. Build prompt
        let prompt = self.build_prompt(query, &bullets);

        // 3. Generate SQL with Phi
        let input = InferenceInput::Text { prompt };
        let output = runtime.infer_sync(model_id, input)?;

        let generated_text = match output {
            InferenceOutput::Text { content } => content,
            InferenceOutput::Boxes { .. } => return Err(anyhow!("Unexpected output type: Boxes")),
            InferenceOutput::Json { .. } => return Err(anyhow!("Unexpected output type: Json")),
        };

        // 4. Extract SQL from response
        let sql = Self::extract_sql(&generated_text);

        // 5. Validate and execute SQL
        let results = if let Some(ref sql_query) = sql {
            match self.execute_search(cyan_db, sql_query) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("SQL execution failed: {}", e);
                    // Fallback to simple LIKE search
                    self.fallback_search(cyan_db, query).unwrap_or_default()
                }
            }
        } else {
            // No SQL generated, use fallback
            self.fallback_search(cyan_db, query).unwrap_or_default()
        };

        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(LensResponse {
            request_id: request_id.to_string(),
            query: query.to_string(),
            generated_sql: sql,
            results,
            playbook_bullets_used: bullet_ids,
            latency_ms,
            error: None,
        })
    }

    /// Search with pre-fetched bullets (avoids holding MutexGuard across await)
    pub fn search_with_bullets(
        &self,
        runtime: &mut Runtime,
        model_id: &str,
        bullets: &[Bullet],
        cyan_db: &Connection,
        request_id: &str,
        query: &str,
    ) -> Result<LensResponse> {
        let start = std::time::Instant::now();

        let bullet_ids: Vec<String> = bullets.iter().map(|b| b.id.clone()).collect();

        // Build prompt with provided bullets
        let prompt = self.build_prompt(query, bullets);

        // Generate SQL with Phi
        let input = InferenceInput::Text { prompt };
        let output = runtime.infer_sync(model_id, input)?;

        let generated_text = match output {
            InferenceOutput::Text { content } => content,
            InferenceOutput::Boxes { .. } => return Err(anyhow!("Unexpected output type: Boxes")),
            InferenceOutput::Json { .. } => return Err(anyhow!("Unexpected output type: Json")),
        };

        // Extract SQL from response
        let sql = Self::extract_sql(&generated_text);

        // Validate and execute SQL
        let results = if let Some(ref sql_query) = sql {
            match self.execute_search(cyan_db, sql_query) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("SQL execution failed: {}", e);
                    self.fallback_search(cyan_db, query).unwrap_or_default()
                }
            }
        } else {
            self.fallback_search(cyan_db, query).unwrap_or_default()
        };

        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(LensResponse {
            request_id: request_id.to_string(),
            query: query.to_string(),
            generated_sql: sql,
            results,
            playbook_bullets_used: bullet_ids,
            latency_ms,
            error: None,
        })
    }

    /// Build prompt with playbook context
    fn build_prompt(&self, query: &str, bullets: &[Bullet]) -> String {
        let mut prompt = String::new();

        // System context
        prompt.push_str("<|system|>\n");
        prompt.push_str("You are CyanLens, an AI assistant that generates SQL queries to search a workspace database.\n");
        prompt.push_str("Generate ONLY valid SQLite SELECT queries. Do not use INSERT, UPDATE, DELETE, or DROP.\n");
        prompt.push_str("<|end|>\n");

        // Schema context
        prompt.push_str("<|user|>\n");
        prompt.push_str("## Database Schema\n");
        prompt.push_str(&self.schema_context);
        prompt.push('\n');

        // Playbook context
        if !bullets.is_empty() {
            prompt.push_str("## Learned Patterns (from previous corrections)\n");
            for bullet in bullets {
                prompt.push_str(&format!("- {}\n", bullet.content));
            }
            prompt.push('\n');
        }

        // User query
        prompt.push_str(&format!("## User Query\n{}\n\n", query));
        prompt.push_str("Generate a SQL SELECT query to answer this. Output only the SQL, no explanation.\n");
        prompt.push_str("<|end|>\n");

        prompt.push_str("<|assistant|>\n");

        prompt
    }

    /// Extract SQL from model output
    pub(crate) fn extract_sql(response: &str) -> Option<String> {
        let response = response.trim();

        // Try to find SQL in code block
        if let Some(start) = response.find("```sql") {
            let content_start = start + 6;
            if let Some(end) = response[content_start..].find("```") {
                let sql = response[content_start..content_start + end].trim();
                if Self::validate_sql(sql) {
                    return Some(sql.to_string());
                }
            }
        }

        // Try generic code block
        if let Some(start) = response.find("```") {
            let content_start = start + 3;
            // Skip language identifier
            let content_start = response[content_start..]
                .find('\n')
                .map(|i| content_start + i + 1)
                .unwrap_or(content_start);

            if let Some(end) = response[content_start..].find("```") {
                let sql = response[content_start..content_start + end].trim();
                if Self::validate_sql(sql) {
                    return Some(sql.to_string());
                }
            }
        }

        // Try raw response if it looks like SQL
        if response.to_uppercase().starts_with("SELECT") && Self::validate_sql(response) {
            return Some(response.to_string());
        }

        None
    }

    /// Validate SQL is safe (SELECT only)
    fn validate_sql(sql: &str) -> bool {
        let upper = sql.to_uppercase();

        // Must start with SELECT
        if !upper.trim_start().starts_with("SELECT") {
            return false;
        }

        // Block dangerous keywords
        let forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "EXEC", "EXECUTE"];
        for keyword in forbidden {
            if upper.contains(keyword) {
                return false;
            }
        }

        // Block multiple statements
        if sql.matches(';').count() > 1 {
            return false;
        }

        true
    }

    /// Execute search SQL and format results
    pub fn execute_search(&self, db: &Connection, sql: &str) -> Result<Vec<SearchResult>> {
        // Add LIMIT if not present
        let sql = if !sql.to_uppercase().contains("LIMIT") {
            format!("{} LIMIT 100", sql.trim_end_matches(';'))
        } else {
            sql.to_string()
        };

        let mut stmt = db.prepare(&sql)?;
        let column_names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();

        let mut results = Vec::new();
        let mut rows = stmt.query([])?;

        while let Some(row) = rows.next()? {
            let result = self.row_to_result(row, &column_names)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Convert a row to SearchResult
    fn row_to_result(&self, row: &rusqlite::Row, columns: &[String]) -> Result<SearchResult> {
        // Try to extract common fields
        let id = self.get_string_column(row, columns, &["id", "board_id", "object_id"])
            .unwrap_or_else(|| "unknown".to_string());

        let name = self.get_string_column(row, columns, &["name", "title"])
            .unwrap_or_else(|| id.clone());

        let result_type = self.get_string_column(row, columns, &["type", "cell_type"])
            .unwrap_or_else(|| "board".to_string());

        let snippet = self.get_string_column(row, columns, &["content", "snippet", "description"]);

        // Build deep link
        let workspace_id = self.get_string_column(row, columns, &["workspace_id"]);
        let group_id = self.get_string_column(row, columns, &["group_id"]);

        let deep_link = self.build_deep_link(&result_type, &id, workspace_id.as_deref(), group_id.as_deref());

        // Collect all columns as metadata
        let mut metadata = serde_json::Map::new();
        for (i, col) in columns.iter().enumerate() {
            if let Ok(val) = row.get::<_, String>(i) {
                metadata.insert(col.clone(), serde_json::Value::String(val));
            } else if let Ok(val) = row.get::<_, i64>(i) {
                metadata.insert(col.clone(), serde_json::Value::Number(val.into()));
            }
        }

        Ok(SearchResult {
            id,
            name,
            result_type,
            snippet,
            deep_link,
            metadata: Some(serde_json::Value::Object(metadata)),
        })
    }

    fn get_string_column(&self, row: &rusqlite::Row, columns: &[String], candidates: &[&str]) -> Option<String> {
        for candidate in candidates {
            if let Some(idx) = columns.iter().position(|c| c == *candidate) {
                if let Ok(val) = row.get::<_, String>(idx) {
                    return Some(val);
                }
            }
        }
        None
    }

    /// Build deep link URL
    fn build_deep_link(&self, result_type: &str, id: &str, workspace_id: Option<&str>, group_id: Option<&str>) -> String {
        match result_type.to_lowercase().as_str() {
            "board" | "whiteboard" | "notebook" => {
                if let (Some(gid), Some(wid)) = (group_id, workspace_id) {
                    format!("cyan://group/{}/workspace/{}/board/{}", gid, wid, id)
                } else if let Some(wid) = workspace_id {
                    format!("cyan://workspace/{}/board/{}", wid, id)
                } else {
                    format!("cyan://board/{}", id)
                }
            }
            "workspace" => {
                if let Some(gid) = group_id {
                    format!("cyan://group/{}/workspace/{}", gid, id)
                } else {
                    format!("cyan://workspace/{}", id)
                }
            }
            "group" => format!("cyan://group/{}", id),
            "cell" => {
                // cells need board_id
                format!("cyan://cell/{}", id)
            }
            _ => format!("cyan://item/{}", id),
        }
    }

    /// Fallback search using simple LIKE
    fn fallback_search(&self, db: &Connection, query: &str) -> Result<Vec<SearchResult>> {
        let pattern = format!("%{}%", query);

        let mut stmt = db.prepare(
            "SELECT o.id, o.name, o.type, w.id as workspace_id, g.id as group_id
             FROM objects o
             LEFT JOIN workspaces w ON o.workspace_id = w.id
             LEFT JOIN groups g ON w.group_id = g.id
             WHERE o.name LIKE ?1
             ORDER BY o.last_accessed DESC
             LIMIT 20"
        )?;

        let results = stmt.query_map(params![pattern], |row| {
            let id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let result_type: String = row.get(2)?;
            let workspace_id: Option<String> = row.get(3)?;
            let group_id: Option<String> = row.get(4)?;

            let deep_link = if let (Some(gid), Some(wid)) = (&group_id, &workspace_id) {
                format!("cyan://group/{}/workspace/{}/board/{}", gid, wid, id)
            } else {
                format!("cyan://board/{}", id)
            };

            Ok(SearchResult {
                id,
                name,
                result_type,
                snippet: None,
                deep_link,
                metadata: None,
            })
        })?.filter_map(|r| r.ok()).collect();

        Ok(results)
    }

    /// Process user feedback
    pub fn process_feedback(
        &self,
        playbook_db: &Connection,
        feedback: &LensFeedback,
    ) -> Result<Option<String>> {
        // Update bullet feedback
        for bf in &feedback.bullet_feedback {
            let tag = FeedbackTag::from_str(&bf.tag);
            if let Err(e) = playbook::record_feedback(playbook_db, &bf.bullet_id, tag) {
                tracing::warn!("Failed to record feedback for {}: {}", bf.bullet_id, e);
            }
        }

        // Create new bullet from correction
        if let Some(ref correction) = feedback.correction {
            let content = if correction.correct_sql.is_some() {
                format!(
                    "When user asks '{}', the correct approach is: {}",
                    correction.wrong_sql, correction.explanation
                )
            } else {
                format!(
                    "Avoid: {}. Instead: {}",
                    correction.wrong_sql, correction.explanation
                )
            };

            let section = if feedback.was_helpful {
                Section::Strategies
            } else {
                Section::Mistakes
            };

            let bullet_id = playbook::add_with_source(
                playbook_db,
                &self.scope,
                section,
                &content,
                "lens_feedback",
                &feedback.request_id,
            )?;

            return Ok(Some(bullet_id));
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_sql() {
        assert!(CyanLens::validate_sql("SELECT * FROM objects"));
        assert!(CyanLens::validate_sql("SELECT id, name FROM objects WHERE name LIKE '%test%'"));
        assert!(CyanLens::validate_sql("  SELECT * FROM objects LIMIT 10"));

        assert!(!CyanLens::validate_sql("DELETE FROM objects"));
        assert!(!CyanLens::validate_sql("INSERT INTO objects VALUES (1)"));
        assert!(!CyanLens::validate_sql("DROP TABLE objects"));
        assert!(!CyanLens::validate_sql("SELECT * FROM objects; DROP TABLE objects"));
    }

    #[test]
    fn test_extract_sql() {
        // Code block
        let response = "Here's the query:\n```sql\nSELECT * FROM objects\n```";
        assert_eq!(
            CyanLens::extract_sql(response),
            Some("SELECT * FROM objects".to_string())
        );

        // Raw SQL
        let response = "SELECT id, name FROM objects WHERE name LIKE '%test%'";
        assert_eq!(
            CyanLens::extract_sql(response),
            Some(response.to_string())
        );

        // Invalid
        let response = "I don't understand the query";
        assert_eq!(CyanLens::extract_sql(response), None);
    }

    #[test]
    fn test_build_deep_link() {
        let lens = CyanLens::new("test");

        assert_eq!(
            lens.build_deep_link("board", "b123", Some("w456"), Some("g789")),
            "cyan://group/g789/workspace/w456/board/b123"
        );

        assert_eq!(
            lens.build_deep_link("workspace", "w456", None, Some("g789")),
            "cyan://group/g789/workspace/w456"
        );

        assert_eq!(
            lens.build_deep_link("group", "g789", None, None),
            "cyan://group/g789"
        );
    }
}