//! SKILL.md parsing and model manifest types
//!
//! Each model bundle contains a SKILL.md file with YAML frontmatter
//! describing capabilities, input/output schemas, and metadata.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Model capability - what the model can do
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Text generation (chat, completion)
    TextGeneration,
    /// Generate Mermaid diagrams from text
    TextToMermaid,
    /// Generate Markdown documents
    TextToMarkdown,
    /// Detect objects/shapes in images (YOLO)
    ImageToBoxes,
    /// Extract text from images (OCR)
    ImageToText,
    /// Analyze project health from integration data
    ProjectHealth,
    /// Design pattern recommendations
    DesignPatterns,
    /// Custom capability
    Custom(String),
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
    /// Supported formats (e.g., ["png", "jpeg"] for images)
    #[serde(default)]
    pub formats: Vec<String>,
    /// JSON schema for structured output (optional)
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
}

/// Parsed SKILL.md manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Model name (unique identifier)
    pub name: String,
    /// Semantic version
    pub version: String,
    /// Runtime kind
    pub kind: ModelKind,
    /// Searchable tags
    #[serde(default)]
    pub tags: Vec<String>,
    /// What this model can do
    pub capabilities: Vec<Capability>,
    /// Input schema
    pub input: IOSchema,
    /// Output schema
    pub output: IOSchema,
    /// Base model name (for LoRA adapters)
    #[serde(default)]
    pub base_model: Option<String>,
    /// LoRA rank (for LoRA adapters)
    #[serde(default)]
    pub lora_rank: Option<u8>,
    /// Author identifier
    #[serde(default)]
    pub author: String,
    /// Creation timestamp (Unix epoch)
    #[serde(default)]
    pub created: i64,
    /// Model file name (relative to SKILL.md directory)
    #[serde(default)]
    pub model_file: Option<String>,
    /// Description (from markdown body)
    #[serde(skip)]
    pub description: String,
    /// Directory containing this skill
    #[serde(skip)]
    pub directory: std::path::PathBuf,
}

impl Skill {
    /// Load a skill from a directory containing SKILL.md
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
        // Split frontmatter from body
        let (frontmatter, body) = Self::split_frontmatter(content)?;

        // Parse YAML frontmatter
        let mut skill: Skill = serde_yaml::from_str(&frontmatter)
            .map_err(|e| anyhow!("Failed to parse SKILL.md frontmatter: {}", e))?;

        // Extract description from markdown body
        skill.description = body.trim().to_string();

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
    fn detect_model_file(dir: &Path, kind: &ModelKind) -> Option<String> {
        let extensions = match kind {
            ModelKind::Gguf => vec!["gguf"],
            ModelKind::Onnx => vec!["onnx"],
            ModelKind::Lora => vec!["safetensors", "bin"],
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
    pub fn has_capability(&self, cap: &Capability) -> bool {
        self.capabilities.contains(cap)
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

impl Default for IOSchema {
    fn default() -> Self {
        Self {
            io_type: IOType::Text,
            formats: Vec::new(),
            schema: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SKILL: &str = r#"---
name: whiteboard-detector
version: 0.1.0
kind: onnx
tags: [vision, detection, whiteboard]
capabilities:
  - image_to_boxes
input:
  type: image
  formats: [png, jpeg]
output:
  type: boxes
  schema:
    class: string
    confidence: float
    bbox: [x, y, w, h]
author: cyan
created: 1702400000
---

# Whiteboard Shape Detector

Detects 30 shape classes from whiteboard photos.

## Usage

Pass an image, get bounding boxes.
"#;

    #[test]
    fn test_parse_skill() {
        let skill = Skill::parse(SAMPLE_SKILL).unwrap();
        assert_eq!(skill.name, "whiteboard-detector");
        assert_eq!(skill.version, "0.1.0");
        assert_eq!(skill.kind, ModelKind::Onnx);
        assert!(skill.has_capability(&Capability::ImageToBoxes));
        assert_eq!(skill.author, "cyan");
        assert!(skill.description.contains("Whiteboard Shape Detector"));
    }

    #[test]
    fn test_missing_frontmatter() {
        let result = Skill::parse("# No frontmatter here");
        assert!(result.is_err());
    }
}
