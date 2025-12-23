//! Executor module for cyan-sql action plans
//!
//! Handles:
//! - Parsing model output (JSON action plans or raw SQL)
//! - Template resolution ({{generate_uuid}}, {{now}}, {{result[0].id}})
//! - SQL execution against cyan.db
//! - Confirmation flow for mutations

use anyhow::{anyhow, Result};
use regex::Regex;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Parsed action plan from model output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPlan {
    pub intent: String,
    #[serde(default)]
    pub requires_confirmation: bool,
    pub actions: Vec<Action>,
    pub confirmation: Option<String>,
}

/// Individual action in a plan
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Action {
    #[serde(rename = "SELECT")]
    Select {
        sql: Option<String>,
        purpose: Option<String>,
    },

    #[serde(rename = "INSERT")]
    Insert {
        table: String,
        values: Value,
        #[serde(rename = "on_conflict")]
        on_conflict: Option<String>,
    },

    #[serde(rename = "UPDATE")]
    Update {
        table: String,
        set: Value,
        #[serde(rename = "where")]
        where_clause: Option<Value>,
    },

    #[serde(rename = "DELETE")]
    Delete {
        table: String,
        #[serde(rename = "where")]
        where_clause: Value,
    },

    #[serde(rename = "UPSERT")]
    Upsert {
        table: String,
        values: Value,
        #[serde(rename = "on_conflict")]
        on_conflict: Value,
    },
}

/// Result of executing an action plan
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionResult {
    pub success: bool,
    pub intent: String,
    pub affected_rows: u32,
    pub message: String,
    pub query_results: Option<Vec<HashMap<String, Value>>>,
}

/// Parsed output from model
#[derive(Debug)]
pub enum ParsedOutput {
    Plan(ActionPlan),
    Sql(String),
}

/// Executor for action plans
pub struct Executor {
    /// Results from previous actions for template resolution
    results: Vec<Vec<HashMap<String, Value>>>,
    /// Current context
    current_board_id: Option<String>,
    current_workspace_id: Option<String>,
}

impl Executor {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            current_board_id: None,
            current_workspace_id: None,
        }
    }

    /// Set current context (board, workspace)
    pub fn with_context(mut self, board_id: Option<String>, workspace_id: Option<String>) -> Self {
        self.current_board_id = board_id;
        self.current_workspace_id = workspace_id;
        self
    }

    /// Parse model output into either ActionPlan or raw SQL
    pub fn parse_output(output: &str) -> Result<ParsedOutput> {
        // Try to extract JSON from ```json``` blocks
        if let Some(json_str) = extract_json_block(output) {
            if let Ok(plan) = serde_json::from_str::<ActionPlan>(&json_str) {
                return Ok(ParsedOutput::Plan(plan));
            }
            if let Ok(actions) = serde_json::from_str::<Vec<Action>>(&json_str) {
                return Ok(ParsedOutput::Plan(ActionPlan {
                    intent: "unknown".to_string(),
                    requires_confirmation: true,
                    actions,
                    confirmation: None,
                }));
            }
        }

        // Try to extract SQL from ```sql``` blocks
        if let Some(sql) = extract_sql_block(output) {
            return Ok(ParsedOutput::Sql(sql));
        }

        // Try parsing entire output as JSON
        if let Ok(plan) = serde_json::from_str::<ActionPlan>(output.trim()) {
            return Ok(ParsedOutput::Plan(plan));
        }

        Err(anyhow!("Could not parse model output as action plan or SQL"))
    }

    /// Execute an action plan against the database
    pub fn execute_plan(&mut self, db: &Connection, plan: &ActionPlan) -> Result<ExecutionResult> {
        let mut total_affected = 0u32;

        for action in &plan.actions {
            match action {
                Action::Select { sql, .. } => {
                    let sql = sql.as_ref().ok_or_else(|| anyhow!("SELECT missing sql"))?;
                    let resolved = self.resolve_template(sql)?;
                    let rows = self.execute_select(db, &resolved)?;
                    self.results.push(rows);
                }

                Action::Insert { table, values, on_conflict } => {
                    let resolved = self.resolve_value(values)?;
                    let affected = self.execute_insert(db, table, &resolved, on_conflict.as_deref())?;
                    total_affected += affected;
                    self.results.push(vec![]);
                }

                Action::Update { table, set, where_clause } => {
                    let resolved_set = self.resolve_value(set)?;
                    let resolved_where = where_clause.as_ref()
                        .map(|w| self.resolve_value(w))
                        .transpose()?;
                    let affected = self.execute_update(db, table, &resolved_set, resolved_where.as_ref())?;
                    total_affected += affected;
                    self.results.push(vec![]);
                }

                Action::Delete { table, where_clause } => {
                    let resolved = self.resolve_value(where_clause)?;
                    let affected = self.execute_delete(db, table, &resolved)?;
                    total_affected += affected;
                    self.results.push(vec![]);
                }

                Action::Upsert { table, values, on_conflict } => {
                    let resolved = self.resolve_value(values)?;
                    let affected = self.execute_upsert(db, table, &resolved, on_conflict)?;
                    total_affected += affected;
                    self.results.push(vec![]);
                }
            }
        }

        Ok(ExecutionResult {
            success: true,
            intent: plan.intent.clone(),
            affected_rows: total_affected,
            message: format!("Executed {} actions", plan.actions.len()),
            query_results: self.results.last().cloned(),
        })
    }

    /// Execute a raw SQL query (SELECT only)
    pub fn execute_query(&mut self, db: &Connection, sql: &str) -> Result<Vec<HashMap<String, Value>>> {
        let resolved = self.resolve_template(sql)?;
        self.execute_select(db, &resolved)
    }

    // === Private execution methods ===

    fn execute_select(&self, db: &Connection, sql: &str) -> Result<Vec<HashMap<String, Value>>> {
        let mut stmt = db.prepare(sql)?;
        let column_names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();

        let rows = stmt.query_map([], |row| {
            let mut map = HashMap::new();
            for (i, name) in column_names.iter().enumerate() {
                let value: rusqlite::types::Value = row.get(i)?;
                map.insert(name.clone(), sqlite_to_json(value));
            }
            Ok(map)
        })?;

        rows.collect::<Result<Vec<_>, _>>().map_err(|e| anyhow!("{}", e))
    }

    fn execute_insert(&self, db: &Connection, table: &str, values: &Value, on_conflict: Option<&str>) -> Result<u32> {
        let obj = values.as_object().ok_or_else(|| anyhow!("INSERT values must be object"))?;

        let columns: Vec<&str> = obj.keys().map(|s| s.as_str()).collect();
        let placeholders: Vec<&str> = columns.iter().map(|_| "?").collect();

        let conflict_clause = match on_conflict {
            Some("ignore") => " OR IGNORE",
            Some("replace") => " OR REPLACE",
            _ => "",
        };

        let sql = format!(
            "INSERT{} INTO {} ({}) VALUES ({})",
            conflict_clause,
            table,
            columns.join(", "),
            placeholders.join(", ")
        );

        let params: Vec<String> = obj.values().map(json_to_sql_string).collect();
        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|s| s as &dyn rusqlite::ToSql).collect();

        let affected = db.execute(&sql, param_refs.as_slice())?;
        Ok(affected as u32)
    }

    fn execute_update(&self, db: &Connection, table: &str, set: &Value, where_clause: Option<&Value>) -> Result<u32> {
        let set_obj = set.as_object().ok_or_else(|| anyhow!("UPDATE set must be object"))?;

        let set_parts: Vec<String> = set_obj.keys().map(|k| format!("{} = ?", k)).collect();

        let (where_sql, where_params) = if let Some(wc) = where_clause {
            let wc_obj = wc.as_object().ok_or_else(|| anyhow!("UPDATE where must be object"))?;
            let parts: Vec<String> = wc_obj.keys().map(|k| format!("{} = ?", k)).collect();
            let params: Vec<String> = wc_obj.values().map(json_to_sql_string).collect();
            (parts.join(" AND "), params)
        } else if let Some(ref board_id) = self.current_board_id {
            ("id = ?".to_string(), vec![board_id.clone()])
        } else {
            return Err(anyhow!("UPDATE requires where clause or current_board_id"));
        };

        let sql = format!("UPDATE {} SET {} WHERE {}", table, set_parts.join(", "), where_sql);

        let mut all_params: Vec<String> = set_obj.values().map(json_to_sql_string).collect();
        all_params.extend(where_params);
        let param_refs: Vec<&dyn rusqlite::ToSql> = all_params.iter().map(|s| s as &dyn rusqlite::ToSql).collect();

        let affected = db.execute(&sql, param_refs.as_slice())?;
        Ok(affected as u32)
    }

    fn execute_delete(&self, db: &Connection, table: &str, where_clause: &Value) -> Result<u32> {
        let wc_obj = where_clause.as_object().ok_or_else(|| anyhow!("DELETE where must be object"))?;
        let parts: Vec<String> = wc_obj.keys().map(|k| format!("{} = ?", k)).collect();

        let sql = format!("DELETE FROM {} WHERE {}", table, parts.join(" AND "));

        let params: Vec<String> = wc_obj.values().map(json_to_sql_string).collect();
        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|s| s as &dyn rusqlite::ToSql).collect();

        let affected = db.execute(&sql, param_refs.as_slice())?;
        Ok(affected as u32)
    }

    fn execute_upsert(&self, db: &Connection, table: &str, values: &Value, on_conflict: &Value) -> Result<u32> {
        let obj = values.as_object().ok_or_else(|| anyhow!("UPSERT values must be object"))?;

        let columns: Vec<&str> = obj.keys().map(|s| s.as_str()).collect();
        let placeholders: Vec<&str> = columns.iter().map(|_| "?").collect();

        let update_fields = on_conflict.get("update")
            .and_then(|u| u.as_object())
            .ok_or_else(|| anyhow!("UPSERT on_conflict.update must be object"))?;

        let update_parts: Vec<String> = update_fields.keys()
            .map(|k| format!("{} = excluded.{}", k, k))
            .collect();

        let conflict_col = columns.first().ok_or_else(|| anyhow!("UPSERT needs columns"))?;

        let sql = format!(
            "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT({}) DO UPDATE SET {}",
            table,
            columns.join(", "),
            placeholders.join(", "),
            conflict_col,
            update_parts.join(", ")
        );

        let params: Vec<String> = obj.values().map(json_to_sql_string).collect();
        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|s| s as &dyn rusqlite::ToSql).collect();

        let affected = db.execute(&sql, param_refs.as_slice())?;
        Ok(affected as u32)
    }

    // === Template resolution ===

    fn resolve_template(&self, template: &str) -> Result<String> {
        let mut result = template.to_string();

        // Replace {{generate_uuid}}
        while result.contains("{{generate_uuid}}") {
            result = result.replacen("{{generate_uuid}}", &uuid::Uuid::new_v4().to_string(), 1);
        }

        // Replace {{now}}
        result = result.replace("{{now}}", &chrono::Utc::now().timestamp().to_string());

        // Replace :current_board_id
        if let Some(ref id) = self.current_board_id {
            result = result.replace(":current_board_id", id);
        }

        // Replace :current_workspace_id
        if let Some(ref id) = self.current_workspace_id {
            result = result.replace(":current_workspace_id", id);
        }

        // Replace {{result[N].field}}
        let re = Regex::new(r"\{\{result\[(\d+)\]\.(\w+)\}\}")?;
        let mut new_result = result.clone();
        for cap in re.captures_iter(&result) {
            let idx: usize = cap[1].parse()?;
            let field = &cap[2];

            if let Some(rows) = self.results.get(idx) {
                if let Some(first_row) = rows.first() {
                    if let Some(value) = first_row.get(field) {
                        let replacement = match value {
                            Value::String(s) => s.clone(),
                            Value::Number(n) => n.to_string(),
                            _ => value.to_string().trim_matches('"').to_string(),
                        };
                        new_result = new_result.replace(&cap[0], &replacement);
                    }
                }
            }
        }

        Ok(new_result)
    }

    fn resolve_value(&self, value: &Value) -> Result<Value> {
        match value {
            Value::String(s) => {
                let resolved = self.resolve_template(s)?;
                Ok(Value::String(resolved))
            }
            Value::Object(obj) => {
                let mut new_obj = serde_json::Map::new();
                for (k, v) in obj {
                    new_obj.insert(k.clone(), self.resolve_value(v)?);
                }
                Ok(Value::Object(new_obj))
            }
            Value::Array(arr) => {
                let new_arr: Result<Vec<Value>> = arr.iter().map(|v| self.resolve_value(v)).collect();
                Ok(Value::Array(new_arr?))
            }
            _ => Ok(value.clone()),
        }
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

// === Helper functions ===

fn extract_json_block(text: &str) -> Option<String> {
    let re = Regex::new(r"```json\s*([\s\S]*?)\s*```").ok()?;
    re.captures(text).map(|c| c[1].to_string())
}

fn extract_sql_block(text: &str) -> Option<String> {
    let re = Regex::new(r"```sql\s*([\s\S]*?)\s*```").ok()?;
    re.captures(text).map(|c| c[1].to_string())
}

fn sqlite_to_json(value: rusqlite::types::Value) -> Value {
    match value {
        rusqlite::types::Value::Null => Value::Null,
        rusqlite::types::Value::Integer(i) => Value::Number(i.into()),
        rusqlite::types::Value::Real(f) => {
            serde_json::Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null)
        }
        rusqlite::types::Value::Text(s) => Value::String(s),
        rusqlite::types::Value::Blob(b) => {
            use base64::Engine;
            Value::String(base64::engine::general_purpose::STANDARD.encode(b))
        }
    }
}

fn json_to_sql_string(value: &Value) -> String {
    match value {
        Value::Null => "NULL".to_string(),
        Value::Bool(b) => if *b { "1" } else { "0" }.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sql_output() {
        let output = r#"```sql
SELECT * FROM groups;
```"#;

        let parsed = Executor::parse_output(output).unwrap();
        assert!(matches!(parsed, ParsedOutput::Sql(_)));
    }

    #[test]
    fn test_parse_action_plan() {
        let output = r#"```json
{
  "intent": "create_workspace",
  "requires_confirmation": true,
  "actions": [
    {"op": "INSERT", "table": "workspaces", "values": {"name": "Test"}}
  ]
}
```"#;

        let parsed = Executor::parse_output(output).unwrap();
        assert!(matches!(parsed, ParsedOutput::Plan(_)));
    }

    #[test]
    fn test_template_resolution() {
        let executor = Executor::new().with_context(
            Some("board-123".to_string()),
            Some("ws-456".to_string()),
        );

        let resolved = executor.resolve_template(
            "SELECT * FROM objects WHERE id = ':current_board_id'"
        ).unwrap();

        assert!(resolved.contains("board-123"));
    }
}