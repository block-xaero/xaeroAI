//! SKILL.md parsing - Agent Skills format with inline tools
//!
//! Each model package contains a SKILL.md file with YAML frontmatter
//! describing capabilities, inline tools, and playbook scope.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Inline tool definition (embedded in SKILL.md)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineTool {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub context: Option<String>,
}

/// Model kind - runtime to use
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    /// GGUF format, runs via llama.cpp
    Gguf,
    /// ONNX format, runs via ort
    Onnx,
    /// LoRA adapter (requires base model)
    Lora,
}

/// Input/Output type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IOType {
    Text,
    Image,
    Json,
    Boxes,
    Mermaid,
    Markdown,
}

/// Input/Output schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOSchema {
    #[serde(rename = "type")]
    pub io_type: IOType,
    #[serde(default)]
    pub formats: Vec<String>,
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
}

impl Default for IOSchema {
    fn default() -> Self {
        Self {
            io_type: IOType::Text,
            formats: Vec::new(),
            schema: None,
        }
    }
}

/// Capability - what the model can do
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    TextGeneration,
    TextToMermaid,
    TextToMarkdown,
    ImageToBoxes,
    ImageToText,
    SemanticSearch,
    SqlGeneration,
    ProjectHealth,
    DesignPatterns,
    Custom(String),
}

/// Parsed SKILL.md
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    // === Required (Agent Skills Standard) ===
    pub name: String,
    pub description: String,

    // === Model Info ===
    #[serde(default, rename = "model_kind")]
    pub kind: Option<ModelKind>,
    #[serde(default)]
    pub model_file: Option<String>,

    // === Optional ===
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub created: Option<i64>,

    // === I/O Schemas ===
    #[serde(default)]
    pub input: IOSchema,
    #[serde(default)]
    pub output: IOSchema,

    // === Capabilities ===
    #[serde(default)]
    pub capabilities: Vec<String>,

    // === Playbook ===
    #[serde(default)]
    pub playbook_scope: Option<String>,

    // === Inline Tools ===
    #[serde(default)]
    pub tools: Vec<InlineTool>,

    // === Base model for LoRA ===
    #[serde(default)]
    pub base_model: Option<String>,

    // === Parsed from body ===
    #[serde(skip)]
    pub instructions: String,

    #[serde(skip)]
    pub directory: std::path::PathBuf,
}

impl Skill {
    /// Load skill from directory containing SKILL.md
    pub fn load(dir: &Path) -> Result<Self> {
        let skill_path = dir.join("SKILL.md");
        if !skill_path.exists() {
            return Err(anyhow!("SKILL.md not found in {:?}", dir));
        }

        let content = std::fs::read_to_string(&skill_path)?;
        let mut skill = Self::parse(&content)?;
        skill.directory = dir.to_path_buf();

        // Auto-detect model file if not specified
        if skill.model_file.is_none() {
            skill.model_file = Self::detect_model_file(dir, &skill.kind);
        }

        Ok(skill)
    }

    /// Parse SKILL.md content (YAML frontmatter + markdown body)
    pub fn parse(content: &str) -> Result<Self> {
        let (frontmatter, body) = Self::split_frontmatter(content)?;

        let mut skill: Skill = serde_yaml::from_str(&frontmatter)
            .map_err(|e| anyhow!("Failed to parse SKILL.md frontmatter: {}", e))?;

        skill.instructions = body.trim().to_string();

        Ok(skill)
    }

    /// Split frontmatter (between ---) from markdown body
    fn split_frontmatter(content: &str) -> Result<(String, String)> {
        let content = content.trim();

        if !content.starts_with("---") {
            return Err(anyhow!("SKILL.md must start with YAML frontmatter (---)"));
        }

        let rest = &content[3..];
        let end = rest.find("---")
            .ok_or_else(|| anyhow!("SKILL.md frontmatter not closed (missing ---)"))?;

        let frontmatter = rest[..end].trim().to_string();
        let body = rest[end + 3..].to_string();

        Ok((frontmatter, body))
    }

    /// Auto-detect model file in directory
    fn detect_model_file(dir: &Path, kind: &Option<ModelKind>) -> Option<String> {
        let extensions = match kind {
            Some(ModelKind::Gguf) => vec!["gguf"],
            Some(ModelKind::Onnx) => vec!["onnx"],
            Some(ModelKind::Lora) => vec!["safetensors", "bin"],
            None => vec!["gguf", "onnx"],
        };

        for entry in std::fs::read_dir(dir).ok()? {
            let entry = entry.ok()?;
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if extensions.contains(&ext.to_str().unwrap_or("")) {
                    return path.file_name()
                        .map(|n| n.to_string_lossy().to_string());
                }
            }
        }

        None
    }

    /// Get full path to model file
    pub fn model_path(&self) -> Option<std::path::PathBuf> {
        self.model_file.as_ref().map(|f| self.directory.join(f))
    }

    /// Check if this model has a specific capability
    pub fn has_capability(&self, cap: &str) -> bool {
        self.capabilities.iter().any(|c| c == cap)
    }

    /// Get tool by name
    pub fn get_tool(&self, name: &str) -> Option<&InlineTool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Build tool context for prompt injection
    pub fn build_tool_context(&self) -> String {
        if self.tools.is_empty() {
            return String::new();
        }

        let mut ctx = String::from("<available_tools>\n");

        for tool in &self.tools {
            ctx.push_str(&format!("\n## {}\n", tool.name));
            ctx.push_str(&format!("{}\n", tool.description));

            if !tool.constraints.is_empty() {
                ctx.push_str("\nConstraints:\n");
                for c in &tool.constraints {
                    ctx.push_str(&format!("- {}\n", c));
                }
            }

            if let Some(ref context) = tool.context {
                ctx.push_str(&format!("\n{}\n", context));
            }
        }

        ctx.push_str("\n</available_tools>\n");
        ctx
    }

    /// Content hash for deduplication (Blake3 of model file)
    pub fn content_hash(&self) -> Result<String> {
        let model_path = self.model_path()
            .ok_or_else(|| anyhow!("No model file"))?;

        let data = std::fs::read(&model_path)?;
        let hash = blake3::hash(&data);
        Ok(hash.to_hex().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SKILL: &str = r#"---
name: cyan-lens
version: 1.0.0
description: AI assistant for Cyan workspace search and navigation
model_kind: gguf
model_file: phi-3-mini-Q4.gguf
capabilities:
  - text_generation
  - semantic_search
  - sql_generation
playbook_scope: cyan-lens
tools:
  - name: sql_query
    description: Execute read-only SQL queries against workspace database
    constraints:
      - SELECT only
      - No DROP/DELETE/INSERT/UPDATE
      - Max 100 results
    context: |
      Tables: groups, workspaces, objects, notebook_cells, board_metadata

  - name: deep_link
    description: Generate cyan:// URLs for navigation
    context: |
      Format: cyan://group/{gid}/workspace/{wid}/board/{bid}
---

# CyanLens

## When to Use
Use when user wants to search boards, find content, or navigate workspace.

## Instructions
1. Parse user query for intent
2. Check playbook for learned patterns
3. Generate SQL if searching
4. Include deep links in results
"#;

    #[test]
    fn test_parse_skill() {
        let skill = Skill::parse(SAMPLE_SKILL).unwrap();
        assert_eq!(skill.name, "cyan-lens");
        assert_eq!(skill.tools.len(), 2);
        assert_eq!(skill.tools[0].name, "sql_query");
        assert_eq!(skill.tools[1].name, "deep_link");
        assert!(skill.playbook_scope.is_some());
        assert!(skill.instructions.contains("CyanLens"));
    }

    #[test]
    fn test_build_tool_context() {
        let skill = Skill::parse(SAMPLE_SKILL).unwrap();
        let ctx = skill.build_tool_context();
        assert!(ctx.contains("<available_tools>"));
        assert!(ctx.contains("sql_query"));
        assert!(ctx.contains("SELECT only"));
    }

    #[test]
    fn test_missing_frontmatter() {
        let result = Skill::parse("# No frontmatter here");
        assert!(result.is_err());
    }
}