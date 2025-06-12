// src/lib.rs
//! XaeroAI: Generalized framework for building distributed AI agent P2P meshes
//!
//! This crate provides:
//! - Distributed AI agent coordination
//! - Nano model inference and training
//! - Integration points for XaeroFlux and XaeroID

use xaeroflux::actors::XaeroEvent;
use xaeroflux::actors::subject::Subject;
use xaeroid::XaeroID;

pub struct XaeroAIModel {
    pub id: XaeroID,
}

pub struct XaeroAISubject {
    pub id: XaeroID,
    pub ai_subject: Subject,
}

pub struct XaeroAIEvent {
    pub id: XaeroID,
    pub underlying: XaeroEvent,
}

// ============================================================================
// Core AI Events
// ============================================================================

pub struct ModelDiscoveryEvent {
    pub model_id: XaeroID,
    pub capabilities: Vec<String>,
    pub model_type: String,
}

pub struct InferenceRequestEvent {
    pub request_id: XaeroID,
    pub requester: XaeroID,
    pub input_data: Vec<u8>,
    pub task_type: String,
}

pub struct InferenceResponseEvent {
    pub request_id: XaeroID,
    pub responder: XaeroID,
    pub output_data: Vec<u8>,
    pub confidence: f32,
}

pub struct LearningUpdateEvent {
    pub model_id: XaeroID,
    pub update_type: String,
    pub parameters: Vec<u8>,
    pub epoch: u64,
}

pub struct TaskCoordinationEvent {
    pub task_id: XaeroID,
    pub coordinator: XaeroID,
    pub participants: Vec<XaeroID>,
    pub task_spec: String,
}

pub struct ModelStateEvent {
    pub model_id: XaeroID,
    pub state: String, // "loading", "ready", "training", "offline"
    pub metadata: Vec<u8>,
}

// ============================================================================
// Core Traits
// ============================================================================

pub trait XaeroAIAgent {
    fn model_id(&self) -> XaeroID;
    fn capabilities(&self) -> Vec<String>;
    fn model_type(&self) -> String;
}

pub trait XaeroAIInference {
    fn can_handle(&self, task_type: &str) -> bool;
    fn estimate_confidence(&self, input: &[u8]) -> f32;
}

pub trait XaeroAILearning {
    fn can_learn_from(&self, event_type: &str) -> bool;
    fn learning_rate(&self) -> f32;
}

pub trait XaeroAICoordination {
    fn can_coordinate(&self, task_type: &str) -> bool;
    fn max_participants(&self) -> usize;
}

pub trait XaeroAIEventHandler {
    fn handle_discovery(&self, event: &ModelDiscoveryEvent);
    fn handle_inference_request(&self, event: &InferenceRequestEvent);
    fn handle_inference_response(&self, event: &InferenceResponseEvent);
    fn handle_learning_update(&self, event: &LearningUpdateEvent);
    fn handle_task_coordination(&self, event: &TaskCoordinationEvent);
    fn handle_model_state(&self, event: &ModelStateEvent);
}

pub trait XaeroAISubjectManager {
    fn subscribe_to_discovery(&self);
    fn subscribe_to_tasks(&self);
    fn subscribe_to_learning(&self);
    fn publish_discovery(&self, event: ModelDiscoveryEvent);
    fn publish_state(&self, event: ModelStateEvent);
}

// ============================================================================
// Model Registry
// ============================================================================

pub struct ModelRegistry {
    pub discovered_models: Vec<XaeroID>,
}

pub struct ModelCapability {
    pub model_id: XaeroID,
    pub capability: String,
    pub confidence_score: f32,
}

pub struct TaskAssignment {
    pub task_id: XaeroID,
    pub assigned_models: Vec<XaeroID>,
    pub task_type: String,
}

// ============================================================================
// Learning System
// ============================================================================

pub struct LearningSession {
    pub session_id: XaeroID,
    pub participants: Vec<XaeroID>,
    pub learning_type: String,
}

pub struct ParameterUpdate {
    pub model_id: XaeroID,
    pub layer_name: String,
    pub delta: Vec<f32>,
}

pub struct FederatedUpdate {
    pub session_id: XaeroID,
    pub aggregated_parameters: Vec<ParameterUpdate>,
    pub round: u64,
}

// ============================================================================
// Coordination System
// ============================================================================

pub struct TaskSpec {
    pub task_id: XaeroID,
    pub task_type: String,
    pub input_requirements: Vec<String>,
    pub output_format: String,
}

pub struct CoordinationStrategy {
    pub strategy_type: String, // "parallel", "sequential", "voting"
    pub timeout_ms: u64,
    pub min_participants: usize,
}

pub struct ConsensusResult {
    pub task_id: XaeroID,
    pub agreed_output: Vec<u8>,
    pub participant_votes: Vec<(XaeroID, f32)>,
}
