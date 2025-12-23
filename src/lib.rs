//! xaeroai - AI runtime for Cyan
//!
//! Components:
//! - skill: SKILL.md parsing (Agent Skills format)
//! - playbook: ACE-style bullets with SQLite + FTS5
//! - lens: CyanLens search with SQL generation
//! - runtime: GGUF/ONNX model loading & inference
//! - pipeline: Whiteboard → Mermaid conversion
//! - executor: Action plan execution for cyan-sql
//!
//! Storage: SQLite (model_registry, corrections, playbook_bullets)
//! Runtime: GGUF (llama-cpp), ONNX (ort)

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

pub mod skill;
pub mod runtime;
pub mod correction;
pub mod registry;
pub mod dictionary;
pub mod pipeline;
pub mod playbook;
pub mod lens;
pub mod arrow_detector;
pub mod router;
pub mod executor;

pub use skill::{Skill, ModelKind, IOSchema, IOType, Capability, InlineTool};
pub use runtime::{Runtime, InferenceInput, InferenceOutput, DetectedBox};
pub use correction::{Correction, CorrectionInputType};
pub use registry::{ModelRecord, ModelRegistry};
pub use dictionary::{Dictionary, DictionaryBuilder, DomainSource};
pub use pipeline::{WhiteboardPipeline, PipelineResult, DetectedShape, BoundingBox, DiagramType};
pub use playbook::{Bullet, Section, FeedbackTag, PlaybookStats};
pub use lens::{CyanLens, LensResponse, LensFeedback, SearchResult};
pub use router::{Specialist, RouteResult};
pub use executor::{Executor, ActionPlan, ParsedOutput, ExecutionResult};

use anyhow::Result;
use once_cell::sync::OnceCell;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::{
    collections::{VecDeque, HashMap},
    ffi::{c_char, CStr, CString},
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tokio::{runtime::Runtime as TokioRuntime, sync::mpsc};

// ---------- Globals ----------
static TOKIO_RT: OnceCell<TokioRuntime> = OnceCell::new();
static AI_SYSTEM: OnceCell<Arc<AISystem>> = OnceCell::new();

// ---------- Commands (Swift → Rust) ----------
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AICommand {
    // === Model Management ===
    RegisterModel {
        board_id: String,
        file_id: String,
        skill_md: String,
    },
    UnregisterModel { model_id: String },
    LoadModel { model_id: String },
    UnloadModel { model_id: String },
    ListModels { board_id: String },
    ListLoadedModels,

    // === Inference ===
    Infer {
        request_id: String,
        model_id: String,
        input_json: String,
    },
    SwapLora {
        base_model_id: String,
        lora_model_id: String,
    },
    ProcessWhiteboard {
        request_id: String,
        image_hash: String,
    },

    // === CyanLens Search ===
    LensSearch {
        request_id: String,
        query: String,
    },
    LensSearchWithContext {
        request_id: String,
        query: String,
        current_board_id: Option<String>,
        current_workspace_id: Option<String>,
    },
    LensFeedback {
        request_id: String,
        was_helpful: bool,
        bullet_feedback: Vec<BulletFeedbackInput>,
        correction: Option<LensCorrectionInput>,
    },

    // === Agent Confirmation ===
    AgentConfirm {
        request_id: String,
        confirmed: bool,
    },

    // === Playbook Management ===
    PlaybookAdd {
        scope: String,
        section: String,
        content: String,
    },
    PlaybookFeedback {
        bullet_id: String,
        tag: String,
    },
    PlaybookStats { scope: String },
    PlaybookList { scope: String },
    PlaybookDelete { bullet_id: String },

    // === Corrections ===
    LogCorrection {
        model_id: String,
        input_type: String,
        input_data: String,
        original: String,
        corrected: String,
    },
    GetPendingCorrections { limit: u32 },
    MarkCorrectionsSynced { correction_ids: Vec<String> },
    MarkCorrectionsDrained { correction_ids: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulletFeedbackInput {
    pub bullet_id: String,
    pub tag: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensCorrectionInput {
    pub wrong_sql: String,
    pub correct_sql: Option<String>,
    pub explanation: String,
}

// ---------- Events (Rust → Swift) ----------
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AIEvent {
    // === Model Events ===
    ModelRegistered {
        model_id: String,
        board_id: String,
        name: String,
        version: String,
        kind: String,
        capabilities: Vec<String>,
    },
    ModelUnregistered { model_id: String, board_id: String },
    ModelLoaded { model_id: String, name: String },
    ModelUnloaded { model_id: String },
    ModelsList { board_id: String, models: Vec<ModelInfo> },
    LoadedModelsList { models: Vec<String> },

    // === Inference Events ===
    InferenceComplete {
        request_id: String,
        model_id: String,
        output_json: String,
        latency_ms: u64,
    },
    InferenceError { request_id: String, model_id: String, error: String },
    LoraSwapped { base_model_id: String, lora_model_id: String },
    WhiteboardProcessed {
        request_id: String,
        mermaid: String,
        diagram_type: String,
        shape_count: u32,
        latency_ms: u64,
    },
    WhiteboardError { request_id: String, error: String },

    // === CyanLens Events ===
    LensSearchComplete {
        request_id: String,
        query: String,
        routed_to: String,
        route_confidence: f32,
        generated_sql: Option<String>,
        results: Vec<SearchResultEvent>,
        playbook_bullets_used: Vec<String>,
        latency_ms: u64,
    },
    LensSearchError { request_id: String, error: String },
    LensFeedbackRecorded { request_id: String, new_bullet_id: Option<String> },

    // === Agent Events ===
    AgentConfirmation {
        request_id: String,
        intent: String,
        confirmation_message: String,
        actions_preview: Vec<String>,
    },
    AgentExecuted {
        request_id: String,
        intent: String,
        affected_rows: u32,
        message: String,
    },
    AgentError {
        request_id: String,
        step: String,
        error: String,
    },

    // === Playbook Events ===
    PlaybookBulletAdded { bullet_id: String, scope: String, section: String },
    PlaybookFeedbackRecorded { bullet_id: String },
    PlaybookStatsResult {
        scope: String,
        total_bullets: usize,
        by_section: std::collections::HashMap<String, usize>,
        avg_score: f64,
    },
    PlaybookListResult { scope: String, bullets: Vec<BulletInfo> },
    PlaybookBulletDeleted { bullet_id: String },

    // === Correction Events ===
    CorrectionSaved { correction_id: String, model_id: String },
    PendingCorrections { corrections: Vec<CorrectionInfo> },

    // === Error ===
    Error { command: String, error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub kind: String,
    pub capabilities: Vec<String>,
    pub loaded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionInfo {
    pub id: String,
    pub model_id: String,
    pub input_type: String,
    pub input_data: String,
    pub original: String,
    pub corrected: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultEvent {
    pub id: String,
    pub name: String,
    pub result_type: String,
    pub snippet: Option<String>,
    pub deep_link: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulletInfo {
    pub id: String,
    pub section: String,
    pub content: String,
    pub helpful_count: u32,
    pub harmful_count: u32,
    pub score: f32,
}

// ---------- Network Events (P2P broadcast) ----------
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AINetworkEvent {
    ModelRegistered {
        model_id: String,
        board_id: String,
        name: String,
        version: String,
        kind: String,
        capabilities: Vec<String>,
        model_hash: String,
        author: String,
    },
    ModelUnregistered { model_id: String, board_id: String },
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
    PlaybookBulletShared {
        bullet_id: String,
        scope: String,
        section: String,
        content: String,
        user_id: String,
    },
}

// ---------- Pending Plan (for confirmation flow) ----------
#[derive(Debug, Clone)]
struct PendingPlan {
    plan: ActionPlan,
    current_board_id: Option<String>,
    current_workspace_id: Option<String>,
}

// ---------- AI System ----------
pub struct AISystem {
    db: Mutex<Connection>,
    cyan_db_path: PathBuf,
    runtime: Mutex<Runtime>,
    lens: CyanLens,
    sql_lens: CyanLens,
    sql_model_id: Mutex<Option<String>>,
    lens_model_id: Mutex<Option<String>>,
    event_queue: Mutex<VecDeque<AIEvent>>,
    network_event_queue: Mutex<VecDeque<AINetworkEvent>>,
    _command_tx: mpsc::UnboundedSender<AICommand>,
    models_dir: PathBuf,
    user_id: String,
    pending_plans: Mutex<HashMap<String, PendingPlan>>,
}

impl AISystem {
    fn new(db_path: &str, cyan_db_path: &str, models_dir: &str, user_id: &str) -> Result<Self> {
        let db = Connection::open(db_path)?;

        // Initialize tables
        registry::init_table(&db)?;
        correction::init_table(&db)?;
        playbook::init_tables(&db)?;

        let runtime = Runtime::new()?;
        let (command_tx, _command_rx) = mpsc::unbounded_channel();
        let lens = CyanLens::new("cyan-lens");
        let sql_lens = CyanLens::new("cyan-sql");

        Ok(Self {
            db: Mutex::new(db),
            cyan_db_path: PathBuf::from(cyan_db_path),
            runtime: Mutex::new(runtime),
            lens,
            sql_lens,
            sql_model_id: Mutex::new(None),
            lens_model_id: Mutex::new(None),
            event_queue: Mutex::new(VecDeque::new()),
            network_event_queue: Mutex::new(VecDeque::new()),
            _command_tx: command_tx,
            models_dir: PathBuf::from(models_dir),
            user_id: user_id.to_string(),
            pending_plans: Mutex::new(HashMap::new()),
        })
    }

    fn push_event(&self, event: AIEvent) {
        self.event_queue.lock().unwrap().push_back(event);
    }

    fn pop_event(&self) -> Option<AIEvent> {
        self.event_queue.lock().unwrap().pop_front()
    }

    fn push_network_event(&self, event: AINetworkEvent) {
        self.network_event_queue.lock().unwrap().push_back(event);
    }

    fn pop_network_event(&self) -> Option<AINetworkEvent> {
        self.network_event_queue.lock().unwrap().pop_front()
    }

    pub fn handle_command(&self, cmd: AICommand) {
        match cmd {
            AICommand::RegisterModel { board_id, file_id, skill_md } =>
                self.handle_register_model(board_id, file_id, skill_md),
            AICommand::UnregisterModel { model_id } =>
                self.handle_unregister_model(model_id),
            AICommand::LoadModel { model_id } =>
                self.handle_load_model(model_id),
            AICommand::UnloadModel { model_id } =>
                self.handle_unload_model(model_id),
            AICommand::ListModels { board_id } =>
                self.handle_list_models(board_id),
            AICommand::ListLoadedModels =>
                self.handle_list_loaded_models(),
            AICommand::Infer { request_id, model_id, input_json } =>
                self.handle_infer(request_id, model_id, input_json),
            AICommand::SwapLora { base_model_id, lora_model_id } =>
                self.handle_swap_lora(base_model_id, lora_model_id),
            AICommand::ProcessWhiteboard { request_id, image_hash } =>
                self.handle_process_whiteboard(request_id, image_hash),
            AICommand::LensSearch { request_id, query } =>
                self.handle_lens_search(request_id, query, None, None),
            AICommand::LensSearchWithContext { request_id, query, current_board_id, current_workspace_id } =>
                self.handle_lens_search(request_id, query, current_board_id, current_workspace_id),
            AICommand::LensFeedback { request_id, was_helpful, bullet_feedback, correction } =>
                self.handle_lens_feedback(request_id, was_helpful, bullet_feedback, correction),
            AICommand::AgentConfirm { request_id, confirmed } =>
                self.handle_agent_confirm(request_id, confirmed),
            AICommand::PlaybookAdd { scope, section, content } =>
                self.handle_playbook_add(scope, section, content),
            AICommand::PlaybookFeedback { bullet_id, tag } =>
                self.handle_playbook_feedback(bullet_id, tag),
            AICommand::PlaybookStats { scope } =>
                self.handle_playbook_stats(scope),
            AICommand::PlaybookList { scope } =>
                self.handle_playbook_list(scope),
            AICommand::PlaybookDelete { bullet_id } =>
                self.handle_playbook_delete(bullet_id),
            AICommand::LogCorrection { model_id, input_type, input_data, original, corrected } =>
                self.handle_log_correction(model_id, input_type, input_data, original, corrected),
            AICommand::GetPendingCorrections { limit } =>
                self.handle_get_pending_corrections(limit),
            AICommand::MarkCorrectionsSynced { correction_ids } =>
                self.handle_mark_corrections_synced(correction_ids),
            AICommand::MarkCorrectionsDrained { correction_ids } =>
                self.handle_mark_corrections_drained(correction_ids),
        }
    }

    // === Model Handlers ===

    fn handle_register_model(&self, board_id: String, file_id: String, skill_md: String) {
        let result = (|| -> Result<ModelRecord> {
            let skill = Skill::parse(&skill_md)?;

            let record = ModelRecord {
                id: uuid::Uuid::new_v4().to_string(),
                board_id: board_id.clone(),
                name: skill.name.clone(),
                version: skill.version.clone().unwrap_or_else(|| "0.0.0".to_string()),
                kind: skill.kind.as_ref().map(|k| format!("{:?}", k).to_lowercase())
                    .unwrap_or_else(|| "unknown".to_string()),
                capabilities: skill.capabilities.clone(),
                tags: vec![],
                skill_md: skill_md.clone(),
                model_hash: String::new(),
                file_id: Some(file_id),
                author: skill.author.clone().unwrap_or_default(),
                created_at: chrono::Utc::now().timestamp(),
                updated_at: chrono::Utc::now().timestamp(),
            };

            let db = self.db.lock().unwrap();
            registry::insert(&db, &record)?;
            Ok(record)
        })();

        match result {
            Ok(record) => {
                self.push_event(AIEvent::ModelRegistered {
                    model_id: record.id.clone(),
                    board_id: record.board_id.clone(),
                    name: record.name.clone(),
                    version: record.version.clone(),
                    kind: record.kind.clone(),
                    capabilities: record.capabilities.clone(),
                });
                self.push_network_event(AINetworkEvent::ModelRegistered {
                    model_id: record.id,
                    board_id: record.board_id,
                    name: record.name,
                    version: record.version,
                    kind: record.kind,
                    capabilities: record.capabilities,
                    model_hash: record.model_hash,
                    author: record.author,
                });
            }
            Err(e) => self.push_event(AIEvent::Error {
                command: "RegisterModel".to_string(),
                error: e.to_string(),
            }),
        }
    }

    fn handle_unregister_model(&self, model_id: String) {
        let db = self.db.lock().unwrap();
        if let Ok(Some(record)) = registry::get(&db, &model_id) {
            let _ = registry::delete(&db, &model_id);
            self.push_event(AIEvent::ModelUnregistered {
                model_id: model_id.clone(),
                board_id: record.board_id.clone(),
            });
            self.push_network_event(AINetworkEvent::ModelUnregistered {
                model_id,
                board_id: record.board_id,
            });
        }
    }

    fn handle_load_model(&self, model_id: String) {
        let result = (|| -> Result<String> {
            let db = self.db.lock().unwrap();
            let record = registry::get(&db, &model_id)?
                .ok_or_else(|| anyhow::anyhow!("Model not found"))?;
            drop(db);

            let skill = Skill::parse(&record.skill_md)?;
            let model_dir = self.models_dir.join(&record.name);

            let mut runtime = self.runtime.lock().unwrap();
            runtime.load_from_skill(&skill, &model_dir)?;

            if skill.has_capability("sql_generation") ||
                skill.playbook_scope.as_deref() == Some("cyan-lens") {
                *self.lens_model_id.lock().unwrap() = Some(model_id.clone());
            }

            // Track cyan-sql model separately
            if skill.name == "cyan-sql" ||
                skill.playbook_scope.as_deref() == Some("cyan-sql") {
                *self.sql_model_id.lock().unwrap() = Some(model_id.clone());
            }

            Ok(record.name)
        })();

        match result {
            Ok(name) => self.push_event(AIEvent::ModelLoaded { model_id, name }),
            Err(e) => self.push_event(AIEvent::Error {
                command: "LoadModel".to_string(),
                error: e.to_string(),
            }),
        }
    }

    fn handle_unload_model(&self, model_id: String) {
        self.runtime.lock().unwrap().unload(&model_id);

        let mut lens_id = self.lens_model_id.lock().unwrap();
        if lens_id.as_ref() == Some(&model_id) {
            *lens_id = None;
        }
        drop(lens_id);

        let mut sql_id = self.sql_model_id.lock().unwrap();
        if sql_id.as_ref() == Some(&model_id) {
            *sql_id = None;
        }

        self.push_event(AIEvent::ModelUnloaded { model_id });
    }

    fn handle_list_models(&self, board_id: String) {
        let result = (|| -> Result<Vec<ModelInfo>> {
            let db = self.db.lock().unwrap();
            let records = registry::list_by_board(&db, &board_id)?;
            let runtime = self.runtime.lock().unwrap();

            Ok(records.into_iter().map(|r| ModelInfo {
                id: r.id.clone(),
                name: r.name,
                version: r.version,
                kind: r.kind,
                capabilities: r.capabilities,
                loaded: runtime.is_loaded(&r.id),
            }).collect())
        })();

        match result {
            Ok(models) => self.push_event(AIEvent::ModelsList { board_id, models }),
            Err(e) => self.push_event(AIEvent::Error {
                command: "ListModels".to_string(),
                error: e.to_string(),
            }),
        }
    }

    fn handle_list_loaded_models(&self) {
        let models = self.runtime.lock().unwrap().loaded_models();
        self.push_event(AIEvent::LoadedModelsList { models });
    }

    // === Inference Handlers ===

    fn handle_infer(&self, request_id: String, model_id: String, input_json: String) {
        let result = (|| -> Result<(String, u64)> {
            let input: InferenceInput = serde_json::from_str(&input_json)?;
            let start = std::time::Instant::now();
            let mut runtime = self.runtime.lock().unwrap();
            let output = runtime.infer_sync(&model_id, input)?;
            let latency_ms = start.elapsed().as_millis() as u64;
            Ok((serde_json::to_string(&output)?, latency_ms))
        })();

        match result {
            Ok((output_json, latency_ms)) => self.push_event(AIEvent::InferenceComplete {
                request_id,
                model_id,
                output_json,
                latency_ms,
            }),
            Err(e) => self.push_event(AIEvent::InferenceError {
                request_id,
                model_id,
                error: e.to_string(),
            }),
        }
    }

    fn handle_swap_lora(&self, base_model_id: String, lora_model_id: String) {
        let result = (|| -> Result<()> {
            let db = self.db.lock().unwrap();
            let lora_record = registry::get(&db, &lora_model_id)?
                .ok_or_else(|| anyhow::anyhow!("LoRA model not found"))?;
            drop(db);

            let lora_skill = Skill::parse(&lora_record.skill_md)?;
            let lora_path = lora_skill.model_path()
                .ok_or_else(|| anyhow::anyhow!("LoRA model file not found"))?;

            self.runtime.lock().unwrap().swap_lora(&base_model_id, &lora_path)?;
            Ok(())
        })();

        match result {
            Ok(()) => self.push_event(AIEvent::LoraSwapped { base_model_id, lora_model_id }),
            Err(e) => self.push_event(AIEvent::Error {
                command: "SwapLora".to_string(),
                error: e.to_string(),
            }),
        }
    }

    fn handle_process_whiteboard(&self, request_id: String, _image_hash: String) {
        self.push_event(AIEvent::WhiteboardError {
            request_id,
            error: "Pipeline not yet implemented".to_string(),
        });
    }

    // === CyanLens Handlers ===

    fn handle_lens_search(
        &self,
        request_id: String,
        query: String,
        current_board_id: Option<String>,
        current_workspace_id: Option<String>,
    ) {
        // Route query to appropriate specialist
        let route_result = router::route(&query);

        tracing::info!(
            "🔀 Routing '{}' → {:?} (confidence: {:.0}%, reason: {})",
            query,
            route_result.specialist,
            route_result.confidence * 100.0,
            route_result.reason
        );

        match route_result.specialist {
            router::Specialist::CyanSql => {
                self.handle_sql_search(request_id, query, route_result, current_board_id, current_workspace_id);
            }
            router::Specialist::CyanLens => {
                self.handle_lens_search_internal(request_id, query, route_result);
            }
        }
    }

    fn handle_lens_search_internal(&self, request_id: String, query: String, route_result: RouteResult) {
        let result = (|| -> Result<LensResponse> {
            let model_id = self.lens_model_id.lock().unwrap().clone()
                .ok_or_else(|| anyhow::anyhow!("cyan-lens model not loaded"))?;

            let cyan_db = Connection::open(&self.cyan_db_path)?;
            let playbook_db = self.db.lock().unwrap();
            let mut runtime = self.runtime.lock().unwrap();

            self.lens.search(&mut runtime, &model_id, &playbook_db, &cyan_db, &request_id, &query)
        })();

        match result {
            Ok(response) => {
                let results = response.results.iter().map(|r| SearchResultEvent {
                    id: r.id.clone(),
                    name: r.name.clone(),
                    result_type: r.result_type.clone(),
                    snippet: r.snippet.clone(),
                    deep_link: r.deep_link.clone(),
                }).collect();

                self.push_event(AIEvent::LensSearchComplete {
                    request_id: response.request_id,
                    query: response.query,
                    routed_to: route_result.specialist.model_id().to_string(),
                    route_confidence: route_result.confidence,
                    generated_sql: response.generated_sql,
                    results,
                    playbook_bullets_used: response.playbook_bullets_used,
                    latency_ms: response.latency_ms,
                });
            }
            Err(e) => self.push_event(AIEvent::LensSearchError {
                request_id,
                error: e.to_string(),
            }),
        }
    }

    fn handle_sql_search(
        &self,
        request_id: String,
        query: String,
        route_result: RouteResult,
        current_board_id: Option<String>,
        current_workspace_id: Option<String>,
    ) {
        let request_id_for_error = request_id.clone();  // Clone for error handling outside closure

        let result = (|| -> Result<()> {
            let start = std::time::Instant::now();

            // Get playbook bullets for SQL scope
            let db = self.db.lock().unwrap();
            let bullets = playbook::retrieve(&db, "cyan-sql", &query, 5)
                .unwrap_or_default();
            let bullet_ids: Vec<String> = bullets.iter().map(|b| b.id.clone()).collect();
            drop(db);

            // Build prompt with SQL schema context
            let prompt = self.build_sql_prompt(&query, &bullets);

            // Get model ID - try cyan-sql first, fall back to lens model
            let model_id = self.sql_model_id.lock().unwrap().clone()
                .or_else(|| self.lens_model_id.lock().unwrap().clone())
                .ok_or_else(|| anyhow::anyhow!("No SQL-capable model loaded"))?;

            // Run inference
            let mut runtime = self.runtime.lock().unwrap();
            let input = InferenceInput::Text { prompt };
            let output = runtime.infer_sync(&model_id, input)?;
            drop(runtime);

            let generated_text = match output {
                InferenceOutput::Text { content } => content,
                _ => return Err(anyhow::anyhow!("Unexpected output type")),
            };

            // Parse output using executor
            let parsed = Executor::parse_output(&generated_text)?;

            match parsed {
                ParsedOutput::Sql(sql) => {
                    // Pure SELECT query - execute immediately
                    let cyan_db = Connection::open(&self.cyan_db_path)?;
                    let results = self.sql_lens.execute_search(&cyan_db, &sql).unwrap_or_default();
                    let latency_ms = start.elapsed().as_millis() as u64;

                    let search_results: Vec<SearchResultEvent> = results.iter().map(|r| SearchResultEvent {
                        id: r.id.clone(),
                        name: r.name.clone(),
                        result_type: r.result_type.clone(),
                        snippet: r.snippet.clone(),
                        deep_link: r.deep_link.clone(),
                    }).collect();

                    self.push_event(AIEvent::LensSearchComplete {
                        request_id,
                        query,
                        routed_to: route_result.specialist.model_id().to_string(),
                        route_confidence: route_result.confidence,
                        generated_sql: Some(sql),
                        results: search_results,
                        playbook_bullets_used: bullet_ids,
                        latency_ms,
                    });
                }

                ParsedOutput::Plan(plan) => {
                    // Action plan - check if confirmation needed
                    if plan.requires_confirmation {
                        let actions_preview: Vec<String> = plan.actions
                            .iter()
                            .map(|a| format!("{:?}", a))
                            .collect();

                        // Store pending plan
                        self.pending_plans.lock().unwrap().insert(
                            request_id.clone(),
                            PendingPlan {
                                plan: plan.clone(),
                                current_board_id,
                                current_workspace_id,
                            },
                        );

                        // Ask user for confirmation
                        self.push_event(AIEvent::AgentConfirmation {
                            request_id,
                            intent: plan.intent,
                            confirmation_message: plan.confirmation.unwrap_or_else(|| "Execute this action?".to_string()),
                            actions_preview,
                        });
                    } else {
                        // Execute immediately
                        self.execute_plan_internal(request_id, plan, current_board_id, current_workspace_id);
                    }
                }
            }

            Ok(())
        })();

        if let Err(e) = result {
            self.push_event(AIEvent::AgentError {
                request_id: request_id_for_error,
                step: "generation".to_string(),
                error: e.to_string(),
            });
        }
    }

    fn handle_agent_confirm(&self, request_id: String, confirmed: bool) {
        let pending = self.pending_plans.lock().unwrap().remove(&request_id);

        if let Some(PendingPlan { plan, current_board_id, current_workspace_id }) = pending {
            if confirmed {
                self.execute_plan_internal(request_id, plan, current_board_id, current_workspace_id);
            } else {
                self.push_event(AIEvent::AgentError {
                    request_id,
                    step: "confirmation".to_string(),
                    error: "User cancelled action".to_string(),
                });
            }
        } else {
            self.push_event(AIEvent::AgentError {
                request_id,
                step: "confirmation".to_string(),
                error: "No pending action found".to_string(),
            });
        }
    }

    fn execute_plan_internal(
        &self,
        request_id: String,
        plan: ActionPlan,
        current_board_id: Option<String>,
        current_workspace_id: Option<String>,
    ) {
        let result = (|| -> Result<ExecutionResult> {
            let cyan_db = Connection::open(&self.cyan_db_path)?;
            let mut executor = Executor::new().with_context(current_board_id, current_workspace_id);
            executor.execute_plan(&cyan_db, &plan)
        })();

        match result {
            Ok(exec_result) => {
                self.push_event(AIEvent::AgentExecuted {
                    request_id,
                    intent: exec_result.intent,
                    affected_rows: exec_result.affected_rows,
                    message: exec_result.message,
                });
            }
            Err(e) => {
                self.push_event(AIEvent::AgentError {
                    request_id,
                    step: "execution".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    fn build_sql_prompt(&self, query: &str, bullets: &[Bullet]) -> String {
        let mut prompt = String::new();

        prompt.push_str("<|system|>\n");
        prompt.push_str("You are CyanLens, an AI assistant for Cyan workspace. ");
        prompt.push_str("Generate SQLite queries to search the workspace database.\n\n");
        prompt.push_str("Tables: groups, workspaces, objects, notebook_cells\n");
        prompt.push_str("Key: To find boards in a group, JOIN objects → workspaces → groups.\n");
        prompt.push_str("<|end|>\n");

        prompt.push_str("<|user|>\n");

        // Add playbook bullets if any
        if !bullets.is_empty() {
            prompt.push_str("Learned patterns:\n");
            for bullet in bullets {
                prompt.push_str(&format!("- {}\n", bullet.content));
            }
            prompt.push('\n');
        }

        prompt.push_str(query);
        prompt.push_str("\n<|end|>\n");
        prompt.push_str("<|assistant|>\n");

        prompt
    }

    fn handle_lens_feedback(
        &self,
        request_id: String,
        was_helpful: bool,
        bullet_feedback: Vec<BulletFeedbackInput>,
        correction: Option<LensCorrectionInput>,
    ) {
        let result = (|| -> Result<Option<String>> {
            let db = self.db.lock().unwrap();

            let feedback = lens::LensFeedback {
                request_id: request_id.clone(),
                was_helpful,
                bullet_feedback: bullet_feedback.iter().map(|bf| lens::BulletFeedback {
                    bullet_id: bf.bullet_id.clone(),
                    tag: bf.tag.clone(),
                }).collect(),
                correction: correction.map(|c| lens::LensCorrection {
                    wrong_sql: c.wrong_sql,
                    correct_sql: c.correct_sql,
                    explanation: c.explanation,
                }),
            };

            self.lens.process_feedback(&db, &feedback)
        })();

        match result {
            Ok(new_bullet_id) => self.push_event(AIEvent::LensFeedbackRecorded {
                request_id,
                new_bullet_id,
            }),
            Err(e) => self.push_event(AIEvent::Error {
                command: "LensFeedback".to_string(),
                error: e.to_string(),
            }),
        }
    }

    // === Playbook Handlers ===

    fn handle_playbook_add(&self, scope: String, section: String, content: String) {
        let result = (|| -> Result<String> {
            let db = self.db.lock().unwrap();
            playbook::add(&db, &scope, Section::from_str(&section), &content)
        })();

        match result {
            Ok(bullet_id) => {
                self.push_event(AIEvent::PlaybookBulletAdded {
                    bullet_id: bullet_id.clone(),
                    scope: scope.clone(),
                    section: section.clone(),
                });
                self.push_network_event(AINetworkEvent::PlaybookBulletShared {
                    bullet_id,
                    scope,
                    section,
                    content,
                    user_id: self.user_id.clone(),
                });
            }
            Err(e) => self.push_event(AIEvent::Error {
                command: "PlaybookAdd".to_string(),
                error: e.to_string(),
            }),
        }
    }

    fn handle_playbook_feedback(&self, bullet_id: String, tag: String) {
        let db = self.db.lock().unwrap();
        match playbook::record_feedback(&db, &bullet_id, FeedbackTag::from_str(&tag)) {
            Ok(()) => self.push_event(AIEvent::PlaybookFeedbackRecorded { bullet_id }),
            Err(e) => self.push_event(AIEvent::Error {
                command: "PlaybookFeedback".to_string(),
                error: e.to_string(),
            }),
        }
    }

    fn handle_playbook_stats(&self, scope: String) {
        let db = self.db.lock().unwrap();
        match playbook::stats(&db, &scope) {
            Ok(stats) => self.push_event(AIEvent::PlaybookStatsResult {
                scope,
                total_bullets: stats.total_bullets,
                by_section: stats.by_section,
                avg_score: stats.avg_score,
            }),
            Err(e) => self.push_event(AIEvent::Error {
                command: "PlaybookStats".to_string(),
                error: e.to_string(),
            }),
        }
    }

    fn handle_playbook_list(&self, scope: String) {
        let db = self.db.lock().unwrap();
        match playbook::list_all(&db, &scope) {
            Ok(bullets) => {
                let infos = bullets.iter().map(|b| BulletInfo {
                    id: b.id.clone(),
                    section: b.section.as_str().to_string(),
                    content: b.content.clone(),
                    helpful_count: b.helpful_count,
                    harmful_count: b.harmful_count,
                    score: b.score,
                }).collect();
                self.push_event(AIEvent::PlaybookListResult { scope, bullets: infos });
            }
            Err(e) => self.push_event(AIEvent::Error {
                command: "PlaybookList".to_string(),
                error: e.to_string(),
            }),
        }
    }

    fn handle_playbook_delete(&self, bullet_id: String) {
        let db = self.db.lock().unwrap();
        match playbook::delete(&db, &bullet_id) {
            Ok(()) => self.push_event(AIEvent::PlaybookBulletDeleted { bullet_id }),
            Err(e) => self.push_event(AIEvent::Error {
                command: "PlaybookDelete".to_string(),
                error: e.to_string(),
            }),
        }
    }

    // === Correction Handlers ===

    fn handle_log_correction(
        &self,
        model_id: String,
        input_type: String,
        input_data: String,
        original: String,
        corrected: String,
    ) {
        let result = (|| -> Result<(String, String)> {
            let db = self.db.lock().unwrap();
            let record = registry::get(&db, &model_id)?
                .ok_or_else(|| anyhow::anyhow!("Model not found"))?;

            let correction = Correction {
                id: uuid::Uuid::new_v4().to_string(),
                model_id: model_id.clone(),
                input_type: input_type.parse()?,
                input_data,
                original,
                corrected,
                user_id: self.user_id.clone(),
                timestamp: chrono::Utc::now().timestamp(),
                synced: false,
                drained: false,
            };

            correction::insert(&db, &correction)?;
            Ok((correction.id, record.name))
        })();

        match result {
            Ok((correction_id, model_name)) => {
                self.push_event(AIEvent::CorrectionSaved {
                    correction_id: correction_id.clone(),
                    model_id: model_id.clone(),
                });

                let db = self.db.lock().unwrap();
                if let Ok(Some(c)) = correction::get(&db, &correction_id) {
                    self.push_network_event(AINetworkEvent::CorrectionLogged {
                        correction_id: c.id,
                        model_id: c.model_id,
                        model_name,
                        input_type: format!("{:?}", c.input_type).to_lowercase(),
                        input_data: c.input_data,
                        original: c.original,
                        corrected: c.corrected,
                        user_id: c.user_id,
                        timestamp: c.timestamp,
                    });
                }
            }
            Err(e) => self.push_event(AIEvent::Error {
                command: "LogCorrection".to_string(),
                error: e.to_string(),
            }),
        }
    }

    fn handle_get_pending_corrections(&self, limit: u32) {
        let db = self.db.lock().unwrap();
        match correction::list_pending(&db, limit) {
            Ok(corrections) => {
                let infos = corrections.into_iter().map(|c| CorrectionInfo {
                    id: c.id,
                    model_id: c.model_id,
                    input_type: format!("{:?}", c.input_type).to_lowercase(),
                    input_data: c.input_data,
                    original: c.original,
                    corrected: c.corrected,
                    timestamp: c.timestamp,
                }).collect();
                self.push_event(AIEvent::PendingCorrections { corrections: infos });
            }
            Err(e) => self.push_event(AIEvent::Error {
                command: "GetPendingCorrections".to_string(),
                error: e.to_string(),
            }),
        }
    }

    fn handle_mark_corrections_synced(&self, correction_ids: Vec<String>) {
        let db = self.db.lock().unwrap();
        for id in correction_ids {
            let _ = correction::mark_synced(&db, &id);
        }
    }

    fn handle_mark_corrections_drained(&self, correction_ids: Vec<String>) {
        let db = self.db.lock().unwrap();
        for id in correction_ids {
            let _ = correction::mark_drained(&db, &id);
        }
    }
}

// ---------- FFI Functions ----------

/// Initialize the AI system
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_init(
    db_path: *const c_char,
    cyan_db_path: *const c_char,
    models_dir: *const c_char,
    user_id: *const c_char,
) -> bool {
    let db_path = unsafe { CStr::from_ptr(db_path) }.to_string_lossy();
    let cyan_db_path = unsafe { CStr::from_ptr(cyan_db_path) }.to_string_lossy();
    let models_dir = unsafe { CStr::from_ptr(models_dir) }.to_string_lossy();
    let user_id = unsafe { CStr::from_ptr(user_id) }.to_string_lossy();

    if TOKIO_RT.set(TokioRuntime::new().unwrap()).is_err() {
        return false;
    }

    match AISystem::new(&db_path, &cyan_db_path, &models_dir, &user_id) {
        Ok(system) => AI_SYSTEM.set(Arc::new(system)).is_ok(),
        Err(_) => false,
    }
}

/// Send a command (JSON)
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_command(json: *const c_char) -> bool {
    let Some(system) = AI_SYSTEM.get() else { return false };
    let json_str = unsafe { CStr::from_ptr(json) }.to_string_lossy();

    match serde_json::from_str::<AICommand>(&json_str) {
        Ok(cmd) => { system.handle_command(cmd); true }
        Err(_) => false,
    }
}

/// Poll for events (JSON or null)
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_poll_event() -> *mut c_char {
    AI_SYSTEM.get()
        .and_then(|s| s.pop_event())
        .and_then(|e| serde_json::to_string(&e).ok())
        .and_then(|j| CString::new(j).ok())
        .map(|c| c.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

/// Poll for network events
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_poll_network_event() -> *mut c_char {
    AI_SYSTEM.get()
        .and_then(|s| s.pop_network_event())
        .and_then(|e| serde_json::to_string(&e).ok())
        .and_then(|j| CString::new(j).ok())
        .map(|c| c.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

/// Free string
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_free_string(s: *mut c_char) {
    if !s.is_null() { unsafe { drop(CString::from_raw(s)); } }
}

/// Shutdown
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_shutdown() {}

/// Quick lens search
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_lens_search(query: *const c_char) -> *mut c_char {
    let Some(system) = AI_SYSTEM.get() else { return std::ptr::null_mut() };
    let query_str = unsafe { CStr::from_ptr(query) }.to_string_lossy();
    let request_id = uuid::Uuid::new_v4().to_string();

    system.handle_command(AICommand::LensSearch {
        request_id: request_id.clone(),
        query: query_str.to_string(),
    });

    CString::new(request_id).unwrap().into_raw()
}

/// Quick feedback
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_lens_feedback(request_id: *const c_char, was_helpful: bool) -> bool {
    let Some(system) = AI_SYSTEM.get() else { return false };
    let request_id = unsafe { CStr::from_ptr(request_id) }.to_string_lossy();

    system.handle_command(AICommand::LensFeedback {
        request_id: request_id.to_string(),
        was_helpful,
        bullet_feedback: vec![],
        correction: None,
    });
    true
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lens_search_command() {
        let cmd = AICommand::LensSearch {
            request_id: "test-123".to_string(),
            query: "find design boards".to_string(),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("LensSearch"));
    }

    #[test]
    fn test_playbook_command() {
        let cmd = AICommand::PlaybookAdd {
            scope: "cyan-lens".to_string(),
            section: "strategies".to_string(),
            content: "Search by group name first".to_string(),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let parsed: AICommand = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, AICommand::PlaybookAdd { .. }));
    }
}