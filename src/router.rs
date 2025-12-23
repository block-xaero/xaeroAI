//! Query Router - Routes queries to specialist models
//!
//! Models:
//! - cyan-sql: SQL generation for workspace search
//! - cyan-lens: Mermaid diagrams, health analysis, design patterns
//!
//! Future:
//! - cyan-code: Code generation
//! - cyan-review: PR review

use serde::{Deserialize, Serialize};

/// Available specialist models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Specialist {
    /// SQL generation for workspace search
    CyanSql,
    /// Mermaid diagrams, health analysis, design patterns
    CyanLens,
}

impl Specialist {
    /// Get the model ID used in runtime
    pub fn model_id(&self) -> &'static str {
        match self {
            Specialist::CyanSql => "cyan-sql",
            Specialist::CyanLens => "cyan-lens",
        }
    }

    /// Get the GGUF filename
    pub fn gguf_file(&self) -> &'static str {
        match self {
            Specialist::CyanSql => "cyan-sql-q4.gguf",
            Specialist::CyanLens => "cyan-lens-q4.gguf",
        }
    }

    /// Get the playbook scope for this specialist
    pub fn playbook_scope(&self) -> &'static str {
        match self {
            Specialist::CyanSql => "cyan-sql",
            Specialist::CyanLens => "cyan-lens",
        }
    }
}

/// Query classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteResult {
    pub specialist: Specialist,
    pub confidence: f32,
    pub reason: &'static str,
}

/// Route a query to the appropriate specialist model
pub fn route(query: &str) -> RouteResult {
    let q = query.to_lowercase();
    let words: Vec<&str> = q.split_whitespace().collect();

    // === SQL Model Triggers ===

    // Direct search verbs
    if starts_with_any(&q, &["find ", "show ", "list ", "search ", "get ", "fetch "]) {
        return RouteResult {
            specialist: Specialist::CyanSql,
            confidence: 0.95,
            reason: "search verb",
        };
    }

    // Question patterns about data
    if starts_with_any(&q, &["what ", "which ", "where ", "how many "])
        && contains_any(&q, &["board", "group", "workspace", "notebook", "cell", "diagram", "mermaid"])
    {
        return RouteResult {
            specialist: Specialist::CyanSql,
            confidence: 0.90,
            reason: "question about workspace data",
        };
    }

    // Entity keywords without creation intent
    if contains_any(&q, &["boards", "groups", "workspaces", "notebooks", "cells"])
        && !contains_any(&q, &["create", "make", "generate", "design", "draw"])
    {
        return RouteResult {
            specialist: Specialist::CyanSql,
            confidence: 0.85,
            reason: "entity query",
        };
    }

    // Count queries
    if contains_any(&q, &["how many", "count", "total", "number of"]) {
        return RouteResult {
            specialist: Specialist::CyanSql,
            confidence: 0.90,
            reason: "count query",
        };
    }

    // Recent/latest queries
    if contains_any(&q, &["recent", "latest", "newest", "oldest", "last "])
        && contains_any(&q, &["board", "workspace", "notebook"])
    {
        return RouteResult {
            specialist: Specialist::CyanSql,
            confidence: 0.90,
            reason: "temporal query",
        };
    }

    // === Cyan-Lens Model Triggers ===

    // Diagram creation
    if contains_any(&q, &["create ", "make ", "generate ", "draw ", "design "])
        && contains_any(&q, &["diagram", "flowchart", "sequence", "class", "er ", "state", "mermaid"])
    {
        return RouteResult {
            specialist: Specialist::CyanLens,
            confidence: 0.95,
            reason: "diagram creation",
        };
    }

    // Direct mermaid request
    if contains_any(&q, &["mermaid", "sequencediagram", "classdiagram", "flowchart", "erdiagram"])
        && !contains_any(&q, &["find ", "show ", "list ", "search "])
    {
        return RouteResult {
            specialist: Specialist::CyanLens,
            confidence: 0.95,
            reason: "mermaid request",
        };
    }

    // Health analysis
    if contains_any(&q, &["analyze", "health", "drift", "stale", "scope creep", "over-architect"]) {
        return RouteResult {
            specialist: Specialist::CyanLens,
            confidence: 0.90,
            reason: "health analysis",
        };
    }

    // Ticket/PR analysis
    if contains_any(&q, &["ticket", "jira", "pr ", "pull request", "slack mention"])
        && contains_any(&q, &["analyze", "review", "assess", "check"])
    {
        return RouteResult {
            specialist: Specialist::CyanLens,
            confidence: 0.90,
            reason: "ticket analysis",
        };
    }

    // Design patterns
    if contains_any(&q, &["pattern", "design pattern", "gof", "factory", "singleton", "observer", "adapter"]) {
        return RouteResult {
            specialist: Specialist::CyanLens,
            confidence: 0.85,
            reason: "design pattern",
        };
    }

    // Rust idioms
    if contains_any(&q, &["rust", "trait", "impl ", "borrow", "lifetime", "async ", "tokio"]) {
        return RouteResult {
            specialist: Specialist::CyanLens,
            confidence: 0.80,
            reason: "rust question",
        };
    }

    // Code analysis
    if q.contains("<code>") || q.contains("```") {
        return RouteResult {
            specialist: Specialist::CyanLens,
            confidence: 0.85,
            reason: "code block",
        };
    }

    // === Default ===

    // If query is short and looks like a search
    if words.len() <= 4 && !contains_any(&q, &["create", "make", "generate", "analyze"]) {
        return RouteResult {
            specialist: Specialist::CyanSql,
            confidence: 0.60,
            reason: "short query default to search",
        };
    }

    // Default to cyan-lens for complex/unclear queries
    RouteResult {
        specialist: Specialist::CyanLens,
        confidence: 0.50,
        reason: "default",
    }
}

/// Check if string starts with any of the prefixes
fn starts_with_any(s: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| s.starts_with(p))
}

/// Check if string contains any of the substrings
fn contains_any(s: &str, substrings: &[&str]) -> bool {
    substrings.iter().any(|sub| s.contains(sub))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sql_routes() {
        // Search verbs
        assert_eq!(route("find Design boards").specialist, Specialist::CyanSql);
        assert_eq!(route("show me all groups").specialist, Specialist::CyanSql);
        assert_eq!(route("list workspaces in Engineering").specialist, Specialist::CyanSql);
        assert_eq!(route("search for notebooks").specialist, Specialist::CyanSql);

        // Questions about data
        assert_eq!(route("what boards are in Design?").specialist, Specialist::CyanSql);
        assert_eq!(route("which workspaces have mermaid diagrams").specialist, Specialist::CyanSql);
        assert_eq!(route("how many boards exist").specialist, Specialist::CyanSql);

        // Temporal
        assert_eq!(route("recent boards").specialist, Specialist::CyanSql);
        assert_eq!(route("latest notebooks").specialist, Specialist::CyanSql);
    }

    #[test]
    fn test_lens_routes() {
        // Diagram creation
        assert_eq!(route("create a sequence diagram for login").specialist, Specialist::CyanLens);
        assert_eq!(route("make a flowchart for error handling").specialist, Specialist::CyanLens);
        assert_eq!(route("generate mermaid for OAuth flow").specialist, Specialist::CyanLens);

        // Health analysis
        assert_eq!(route("analyze this ticket's health").specialist, Specialist::CyanLens);
        assert_eq!(route("check for design drift").specialist, Specialist::CyanLens);

        // Patterns
        assert_eq!(route("what design pattern is this").specialist, Specialist::CyanLens);
        assert_eq!(route("explain the factory pattern").specialist, Specialist::CyanLens);
    }

    #[test]
    fn test_edge_cases() {
        // "boards with mermaid" is a search, not creation
        assert_eq!(route("find boards with mermaid").specialist, Specialist::CyanSql);
        assert_eq!(route("show mermaid boards").specialist, Specialist::CyanSql);

        // Short queries default to search
        assert_eq!(route("Design boards").specialist, Specialist::CyanSql);
        assert_eq!(route("Engineering notebooks").specialist, Specialist::CyanSql);
    }

    #[test]
    fn test_confidence() {
        let result = route("find Design boards");
        assert!(result.confidence >= 0.90);

        let result = route("something unclear");
        assert!(result.confidence <= 0.60);
    }
}