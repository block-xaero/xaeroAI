//! xaeroai - AI runtime for Cyan
//!
//! Two-actor architecture matching cyan-backend:
//! - CommandActor: handles FFI commands (load model, infer, correct)
//! - NetworkActor: handles P2P sync via Iroh (corrections, model discovery)
//!
//! Storage: SQLite (model_registry, corrections tables)
//! Runtime: GGUF (llama-cpp), ONNX (ort)

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

pub mod skill;
pub mod runtime;
pub mod correction;
pub mod registry;
pub mod dictionary;
pub mod pipeline;
pub use skill::{Skill, Capability, ModelKind, IOSchema, IOType};
pub use runtime::{Runtime, InferenceInput, InferenceOutput, DetectedBox};
pub use correction::{Correction, CorrectionInputType};
pub use registry::{ModelRecord, ModelRegistry};
pub use dictionary::{Dictionary, DictionaryBuilder, DomainSource};
pub use pipeline::{WhiteboardPipeline, PipelineResult, DetectedShape, BoundingBox, DiagramType};

use anyhow::Result;
use once_cell::sync::OnceCell;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
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
    /// Register a model from a board's file
    RegisterModel {
        board_id: String,
        file_id: String,
        skill_md: String,
    },
    /// Unregister a model
    UnregisterModel {
        model_id: String,
    },
    /// Load model into memory for inference
    LoadModel {
        model_id: String,
    },
    /// Unload model from memory
    UnloadModel {
        model_id: String,
    },
    /// Run inference
    Infer {
        request_id: String,
        model_id: String,
        input_json: String,
    },
    /// Swap LoRA adapter
    SwapLora {
        base_model_id: String,
        lora_model_id: String,
    },
    /// Log a user correction
    LogCorrection {
        model_id: String,
        input_type: String,
        input_data: String,
        original: String,
        corrected: String,
    },
    /// List models for a board
    ListModels {
        board_id: String,
    },
    /// List all loaded models
    ListLoadedModels,
    /// Run whiteboard pipeline (YOLO → TrOCR → Phi)
    ProcessWhiteboard {
        request_id: String,
        image_hash: String,
    },
    /// Get pending corrections (for XaeroFlux sync)
    GetPendingCorrections {
        limit: u32,
    },
    /// Mark corrections as synced
    MarkCorrectionsSynced {
        correction_ids: Vec<String>,
    },
    /// Mark corrections as drained (incorporated into training)
    MarkCorrectionsDrained {
        correction_ids: Vec<String>,
    },
}

// ---------- Events (Rust → Swift) ----------
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AIEvent {
    /// Model registered in registry
    ModelRegistered {
        model_id: String,
        board_id: String,
        name: String,
        version: String,
        kind: String,
        capabilities: Vec<String>,
    },
    /// Model unregistered
    ModelUnregistered {
        model_id: String,
        board_id: String,
    },
    /// Model loaded into memory
    ModelLoaded {
        model_id: String,
        name: String,
    },
    /// Model unloaded from memory
    ModelUnloaded {
        model_id: String,
    },
    /// Inference completed
    InferenceComplete {
        request_id: String,
        model_id: String,
        output_json: String,
        latency_ms: u64,
    },
    /// Inference failed
    InferenceError {
        request_id: String,
        model_id: String,
        error: String,
    },
    /// LoRA swapped
    LoraSwapped {
        base_model_id: String,
        lora_model_id: String,
    },
    /// Correction saved
    CorrectionSaved {
        correction_id: String,
        model_id: String,
    },
    /// Models list response
    ModelsList {
        board_id: String,
        models: Vec<ModelInfo>,
    },
    /// Loaded models list response
    LoadedModelsList {
        models: Vec<String>,
    },
    /// Whiteboard pipeline complete
    WhiteboardProcessed {
        request_id: String,
        mermaid: String,
        diagram_type: String,
        shape_count: u32,
        latency_ms: u64,
    },
    /// Whiteboard pipeline error
    WhiteboardError {
        request_id: String,
        error: String,
    },
    /// Pending corrections response
    PendingCorrections {
        corrections: Vec<CorrectionInfo>,
    },
    /// Error event
    Error {
        command: String,
        error: String,
    },
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

// ---------- Network Events (for cyan-backend to broadcast) ----------
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AINetworkEvent {
    /// Broadcast when model registered (for peer discovery)
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
    /// Broadcast when model unregistered
    ModelUnregistered {
        model_id: String,
        board_id: String,
    },
    /// Broadcast when correction logged (for XaeroFlux collection)
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
}

// ---------- AI System ----------
pub struct AISystem {
    db: Mutex<Connection>,
    runtime: Mutex<Runtime>,
    event_queue: Mutex<VecDeque<AIEvent>>,
    network_event_queue: Mutex<VecDeque<AINetworkEvent>>,
    command_tx: mpsc::UnboundedSender<AICommand>,
    models_dir: PathBuf,
    user_id: String,
}

impl AISystem {
    fn new(db_path: &str, models_dir: &str, user_id: &str) -> Result<Self> {
        let db = Connection::open(db_path)?;
        
        // Initialize tables
        registry::init_table(&db)?;
        correction::init_table(&db)?;
        
        let runtime = Runtime::new()?;
        let (command_tx, _command_rx) = mpsc::unbounded_channel();
        
        Ok(Self {
            db: Mutex::new(db),
            runtime: Mutex::new(runtime),
            event_queue: Mutex::new(VecDeque::new()),
            network_event_queue: Mutex::new(VecDeque::new()),
            command_tx,
            models_dir: PathBuf::from(models_dir),
            user_id: user_id.to_string(),
        })
    }

    fn push_event(&self, event: AIEvent) {
        let mut queue = self.event_queue.lock().unwrap();
        queue.push_back(event);
    }

    fn push_network_event(&self, event: AINetworkEvent) {
        let mut queue = self.network_event_queue.lock().unwrap();
        queue.push_back(event);
    }

    fn pop_event(&self) -> Option<AIEvent> {
        let mut queue = self.event_queue.lock().unwrap();
        queue.pop_front()
    }

    fn pop_network_event(&self) -> Option<AINetworkEvent> {
        let mut queue = self.network_event_queue.lock().unwrap();
        queue.pop_front()
    }

    fn handle_command(&self, cmd: AICommand) {
        match cmd {
            AICommand::RegisterModel { board_id, file_id, skill_md } => {
                self.handle_register_model(board_id, file_id, skill_md);
            }
            AICommand::UnregisterModel { model_id } => {
                self.handle_unregister_model(model_id);
            }
            AICommand::LoadModel { model_id } => {
                self.handle_load_model(model_id);
            }
            AICommand::UnloadModel { model_id } => {
                self.handle_unload_model(model_id);
            }
            AICommand::Infer { request_id, model_id, input_json } => {
                self.handle_infer(request_id, model_id, input_json);
            }
            AICommand::SwapLora { base_model_id, lora_model_id } => {
                self.handle_swap_lora(base_model_id, lora_model_id);
            }
            AICommand::LogCorrection { model_id, input_type, input_data, original, corrected } => {
                self.handle_log_correction(model_id, input_type, input_data, original, corrected);
            }
            AICommand::ListModels { board_id } => {
                self.handle_list_models(board_id);
            }
            AICommand::ListLoadedModels => {
                self.handle_list_loaded_models();
            }
            AICommand::GetPendingCorrections { limit } => {
                self.handle_get_pending_corrections(limit);
            }
            AICommand::MarkCorrectionsSynced { correction_ids } => {
                self.handle_mark_corrections_synced(correction_ids);
            }
            AICommand::MarkCorrectionsDrained { correction_ids } => {
                self.handle_mark_corrections_drained(correction_ids);
            }
            AICommand::ProcessWhiteboard { request_id, image_hash } => {
                self.handle_process_whiteboard(request_id, image_hash);
            }
        }
    }

    fn handle_register_model(&self, board_id: String, file_id: String, skill_md: String) {
        let result = (|| -> Result<ModelRecord> {
            let skill = Skill::parse(&skill_md)?;
            let db = self.db.lock().unwrap();
            
            let record = ModelRecord {
                id: uuid::Uuid::new_v4().to_string(),
                board_id: board_id.clone(),
                name: skill.name.clone(),
                version: skill.version.clone(),
                kind: format!("{:?}", skill.kind).to_lowercase(),
                capabilities: skill.capabilities.iter()
                    .map(|c| format!("{:?}", c).to_lowercase())
                    .collect(),
                tags: skill.tags.clone(),
                skill_md: skill_md.clone(),
                model_hash: String::new(), // TODO: compute from file
                file_id: Some(file_id),
                author: skill.author.clone(),
                created_at: chrono::Utc::now().timestamp(),
                updated_at: chrono::Utc::now().timestamp(),
            };
            
            registry::insert(&db, &record)?;
            Ok(record)
        })();

        match result {
            Ok(record) => {
                // Push local event
                self.push_event(AIEvent::ModelRegistered {
                    model_id: record.id.clone(),
                    board_id: record.board_id.clone(),
                    name: record.name.clone(),
                    version: record.version.clone(),
                    kind: record.kind.clone(),
                    capabilities: record.capabilities.clone(),
                });
                
                // Push network event for broadcast
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
            Err(e) => {
                self.push_event(AIEvent::Error {
                    command: "RegisterModel".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    fn handle_unregister_model(&self, model_id: String) {
        let result = (|| -> Result<String> {
            let db = self.db.lock().unwrap();
            let record = registry::get(&db, &model_id)?
                .ok_or_else(|| anyhow::anyhow!("Model not found"))?;
            let board_id = record.board_id.clone();
            registry::delete(&db, &model_id)?;
            Ok(board_id)
        })();

        match result {
            Ok(board_id) => {
                self.push_event(AIEvent::ModelUnregistered {
                    model_id: model_id.clone(),
                    board_id: board_id.clone(),
                });
                self.push_network_event(AINetworkEvent::ModelUnregistered {
                    model_id,
                    board_id,
                });
            }
            Err(e) => {
                self.push_event(AIEvent::Error {
                    command: "UnregisterModel".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    fn handle_load_model(&self, model_id: String) {
        let result = (|| -> Result<String> {
            let db = self.db.lock().unwrap();
            let record = registry::get(&db, &model_id)?
                .ok_or_else(|| anyhow::anyhow!("Model not found"))?;
            
            let skill = Skill::parse(&record.skill_md)?;
            drop(db);
            
            let mut runtime = self.runtime.lock().unwrap();
            runtime.load_from_skill(&skill, &self.models_dir)?;
            
            Ok(record.name)
        })();

        match result {
            Ok(name) => {
                self.push_event(AIEvent::ModelLoaded {
                    model_id,
                    name,
                });
            }
            Err(e) => {
                self.push_event(AIEvent::Error {
                    command: "LoadModel".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    fn handle_unload_model(&self, model_id: String) {
        let result = (|| -> Result<()> {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.unload(&model_id)?;
            Ok(())
        })();

        match result {
            Ok(()) => {
                self.push_event(AIEvent::ModelUnloaded { model_id });
            }
            Err(e) => {
                self.push_event(AIEvent::Error {
                    command: "UnloadModel".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    fn handle_infer(&self, request_id: String, model_id: String, input_json: String) {
        let result = (|| -> Result<(String, u64)> {
            let input: InferenceInput = serde_json::from_str(&input_json)?;
            let mut runtime = self.runtime.lock().unwrap();
            let start = std::time::Instant::now();
            let output = runtime.infer_sync(&model_id, input)?;
            let latency_ms = start.elapsed().as_millis() as u64;
            let output_json = serde_json::to_string(&output)?;
            Ok((output_json, latency_ms))
        })();

        match result {
            Ok((output_json, latency_ms)) => {
                self.push_event(AIEvent::InferenceComplete {
                    request_id,
                    model_id,
                    output_json,
                    latency_ms,
                });
            }
            Err(e) => {
                self.push_event(AIEvent::InferenceError {
                    request_id,
                    model_id,
                    error: e.to_string(),
                });
            }
        }
    }

    fn handle_swap_lora(&self, base_model_id: String, lora_model_id: String) {
        let result = (|| -> Result<()> {
            let db = self.db.lock().unwrap();
            let lora_record = registry::get(&db, &lora_model_id)?
                .ok_or_else(|| anyhow::anyhow!("LoRA model not found"))?;
            drop(db);
            
            let lora_path = self.models_dir.join(&lora_record.name);
            let mut runtime = self.runtime.lock().unwrap();
            runtime.swap_lora(&base_model_id, &lora_path)?;
            Ok(())
        })();

        match result {
            Ok(()) => {
                self.push_event(AIEvent::LoraSwapped {
                    base_model_id,
                    lora_model_id,
                });
            }
            Err(e) => {
                self.push_event(AIEvent::Error {
                    command: "SwapLora".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

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
                
                // Get correction for network event
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
            Err(e) => {
                self.push_event(AIEvent::Error {
                    command: "LogCorrection".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    fn handle_list_models(&self, board_id: String) {
        let result = (|| -> Result<Vec<ModelInfo>> {
            let db = self.db.lock().unwrap();
            let records = registry::list_by_board(&db, &board_id)?;
            let runtime = self.runtime.lock().unwrap();
            
            let models = records.into_iter().map(|r| {
                ModelInfo {
                    id: r.id.clone(),
                    name: r.name,
                    version: r.version,
                    kind: r.kind,
                    capabilities: r.capabilities,
                    loaded: runtime.is_loaded(&r.id),
                }
            }).collect();
            
            Ok(models)
        })();

        match result {
            Ok(models) => {
                self.push_event(AIEvent::ModelsList { board_id, models });
            }
            Err(e) => {
                self.push_event(AIEvent::Error {
                    command: "ListModels".to_string(),
                    error: e.to_string(),
                });
            }
        }
    }

    fn handle_list_loaded_models(&self) {
        let runtime = self.runtime.lock().unwrap();
        let models = runtime.loaded_models();
        self.push_event(AIEvent::LoadedModelsList { models });
    }

    fn handle_get_pending_corrections(&self, limit: u32) {
        let result = (|| -> Result<Vec<CorrectionInfo>> {
            let db = self.db.lock().unwrap();
            let corrections = correction::list_pending(&db, limit)?;
            
            Ok(corrections.into_iter().map(|c| CorrectionInfo {
                id: c.id,
                model_id: c.model_id,
                input_type: format!("{:?}", c.input_type).to_lowercase(),
                input_data: c.input_data,
                original: c.original,
                corrected: c.corrected,
                timestamp: c.timestamp,
            }).collect())
        })();

        match result {
            Ok(corrections) => {
                self.push_event(AIEvent::PendingCorrections { corrections });
            }
            Err(e) => {
                self.push_event(AIEvent::Error {
                    command: "GetPendingCorrections".to_string(),
                    error: e.to_string(),
                });
            }
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

    fn handle_process_whiteboard(&self, request_id: String, _image_hash: String) {
        // TODO: Implement pipeline processing
        // 1. Load image from blob store using hash
        // 2. Run YOLO detection
        // 3. Run TrOCR on text regions
        // 4. Apply dictionary correction
        // 5. Generate Mermaid with Phi
        
        self.push_event(AIEvent::WhiteboardError {
            request_id,
            error: "Pipeline not yet implemented".to_string(),
        });
    }
}

// ---------- FFI Functions ----------

/// Initialize the AI system
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_init(
    db_path: *const c_char,
    models_dir: *const c_char,
    user_id: *const c_char,
) -> bool {
    let db_path = unsafe { CStr::from_ptr(db_path) }.to_string_lossy();
    let models_dir = unsafe { CStr::from_ptr(models_dir) }.to_string_lossy();
    let user_id = unsafe { CStr::from_ptr(user_id) }.to_string_lossy();

    // Initialize tokio runtime
    let rt = TokioRuntime::new();
    if rt.is_err() {
        return false;
    }
    let _ = TOKIO_RT.set(rt.unwrap());

    // Initialize AI system
    match AISystem::new(&db_path, &models_dir, &user_id) {
        Ok(system) => {
            let _ = AI_SYSTEM.set(Arc::new(system));
            true
        }
        Err(_) => false,
    }
}

/// Send a command to the AI system
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_command(json: *const c_char) -> bool {
    let Some(system) = AI_SYSTEM.get() else {
        return false;
    };

    let json_str = unsafe { CStr::from_ptr(json) }.to_string_lossy();
    
    match serde_json::from_str::<AICommand>(&json_str) {
        Ok(cmd) => {
            system.handle_command(cmd);
            true
        }
        Err(_) => false,
    }
}

/// Poll for AI events (returns JSON or null)
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_poll_event() -> *mut c_char {
    let Some(system) = AI_SYSTEM.get() else {
        return std::ptr::null_mut();
    };

    match system.pop_event() {
        Some(event) => {
            match serde_json::to_string(&event) {
                Ok(json) => CString::new(json).unwrap().into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }
        None => std::ptr::null_mut(),
    }
}

/// Poll for network events (for cyan-backend to broadcast)
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_poll_network_event() -> *mut c_char {
    let Some(system) = AI_SYSTEM.get() else {
        return std::ptr::null_mut();
    };

    match system.pop_network_event() {
        Some(event) => {
            match serde_json::to_string(&event) {
                Ok(json) => CString::new(json).unwrap().into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }
        None => std::ptr::null_mut(),
    }
}

/// Free a string returned by xaero_ai functions
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)); }
    }
}

/// Shutdown the AI system
#[unsafe(no_mangle)]
pub extern "C" fn xaero_ai_shutdown() {
    // Models will be dropped when system is dropped
    // For now, just a placeholder
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_serialization() {
        let cmd = AICommand::LoadModel {
            model_id: "test-123".to_string(),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("LoadModel"));
    }

    #[test]
    fn test_event_serialization() {
        let event = AIEvent::ModelLoaded {
            model_id: "test-123".to_string(),
            name: "test-model".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("ModelLoaded"));
    }
}

pub mod arrow_detector;
