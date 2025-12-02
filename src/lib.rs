use std::path::PathBuf;
use serde::{Deserialize, Serialize};

// ---------- Model Configuration ----------

/// Model type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    DesignAnalyst,
    // Future: CodeReviewer, PatternDetector, etc.
}

/// Configuration for loading a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: ModelType,
    pub gguf_path: PathBuf,
    pub context_size: usize,       // e.g., 4096
    pub gpu_layers: Option<u32>,   // None = CPU only
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::DesignAnalyst,
            gguf_path: PathBuf::from("models/design-analyst-q4.gguf"),
            context_size: 4096,
            gpu_layers: None,
        }
    }
}

// ---------- Analysis Input ----------

/// Source of code/design to analyze
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisSource {
    /// Local filesystem path
    LocalRepo(PathBuf),
    /// GitHub repository (owner/repo)
    GitHubRepo { owner: String, repo: String },
    /// GitHub PR
    GitHubPR { owner: String, repo: String, pr: u64 },
    /// Raw text (design doc, code snippet, mermaid, etc.)
    RawText { content: String, language: Option<Language> },
}

/// Detected or specified language
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language {
    Python,
    Java,
    Go,
    Cpp,
    Rust,
    Unknown,
}

// ---------- Analysis Context (built from source) ----------

/// Parsed context ready for model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisContext {
    pub source: AnalysisSource,
    pub language: Language,
    pub structure: Option<RepoStructure>,
    pub dependency_graph: Option<DependencyGraph>,
    pub raw_content: Option<String>,
}

/// Directory/file structure
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepoStructure {
    pub root: String,
    pub modules: Vec<ModuleInfo>,
    pub file_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    pub path: String,
    pub imports: Vec<String>,
    pub loc: usize,
}

/// Dependency relationships
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DependencyGraph {
    pub edges: Vec<(String, String)>,  // (from, to)
    pub external_deps: Vec<String>,
}

// ---------- Analysis Output ----------

/// Result from design analyst
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub patterns_detected: Vec<PatternMatch>,
    pub issues: Vec<DesignIssue>,
    pub suggestions: Vec<String>,
    pub raw_response: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern: String,           // e.g., "CQRS", "Repository", "Event Sourcing"
    pub confidence: f32,           // 0.0 - 1.0
    pub location: Option<String>,  // Where in the code/design
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignIssue {
    pub severity: IssueSeverity,
    pub category: String,          // e.g., "coupling", "layer_violation", "god_class"
    pub description: String,
    pub location: Option<String>,
}

// ---------- Actor Command (for XaeroFlux integration) ----------

/// Commands the AI actor can receive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AiCommand {
    /// Analyze a source
    Analyze { source: AnalysisSource },
    /// Chat message (for interactive mode)
    Chat { message: String, context: Option<AnalysisContext> },
    /// Reload model
    ReloadModel { config: ModelConfig },
}

/// Events the AI actor emits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AiEvent {
    /// Analysis complete
    AnalysisComplete { result: AnalysisResult },
    /// Chat response
    ChatResponse { message: String },
    /// Error occurred
    Error { message: String },
    /// Model loaded/ready
    ModelReady { model_type: ModelType },
}

// ---------- Placeholder for future implementation ----------

/// Design Analyst - placeholder struct
pub struct DesignAnalyst {
    config: ModelConfig,
    // model: LlamaModel,  // TODO: llama-cpp-2
}

impl DesignAnalyst {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }

    /// Load the model (async for future compatibility)
    pub async fn load(&mut self) -> anyhow::Result<()> {
        // TODO: Load GGUF with llama-cpp-2
        todo!("Model loading not implemented")
    }

    /// Analyze source
    pub async fn analyze(&self, _source: AnalysisSource) -> anyhow::Result<AnalysisResult> {
        // TODO: Build context, generate prompt, run inference
        todo!("Analysis not implemented")
    }

    /// Interactive chat
    pub async fn chat(&self, _message: &str, _context: Option<&AnalysisContext>) -> anyhow::Result<String> {
        // TODO: Run inference with chat context
        todo!("Chat not implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_sane() {
        let cfg = ModelConfig::default();
        assert_eq!(cfg.model_type, ModelType::DesignAnalyst);
        assert_eq!(cfg.context_size, 4096);
    }

    #[test]
    fn analysis_source_serializes() {
        let source = AnalysisSource::LocalRepo(PathBuf::from("~/code/test"));
        let json = serde_json::to_string(&source).unwrap();
        assert!(json.contains("LocalRepo"));
    }

    #[test]
    fn ai_command_serializes() {
        let cmd = AiCommand::Analyze {
            source: AnalysisSource::RawText {
                content: "test".to_string(),
                language: Some(Language::Rust),
            },
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("Analyze"));
    }
}