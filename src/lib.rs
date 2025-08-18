extern crate core;

pub mod model_analyzer;
pub mod storage;
pub mod vectorizer;
pub mod yolo;

use bytemuck::{Pod, Zeroable};
use candle_core::quantized::QuantizedType;
use candle_core::Tensor;
use rkyv::ser::{Allocator, Writer};
use rkyv::Archive;
use rkyv::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const XAERO_AI_EVENT_TYPE_BASE: u32 = 108;
pub const LORA_ADAPTER_CREATION: u32 = 0;

pub trait XaeroAIModelOps {
    fn forward_with_lora(
        &self,
        input: &Tensor,
        user_id: [u8; 32],
    ) -> Result<Tensor, Box<dyn std::error::Error>>;
}
pub struct XaeroAIModelRegistry {
    pub xaero_id: [u8; 32],
    pub models: Vec<XaeroAIModel>,
}

pub struct XaeroAIModel {
    pub xaero_id: [u8; 32],
    pub model_hash: [u8; 32],
    pub arch: XaeroModelArchitecture,
    pub quantization: Box<dyn QuantizedType>, // INT8, INT4, F16, etc.
    pub lora_adapters: BTreeMap<[u8; 32], [u8; 32]>, // user_id -> lora_delta_hash
    pub base_size_bytes: u64,                 // For P2P transfer planning
}

#[repr(C, align(64))]
#[derive(Archive, Serialize, Deserialize, Debug, Copy, Clone)]
pub enum XaeroLayerType {
    Conv,
    Linear,
    Attention,
}

#[repr(C, align(64))]
#[derive(Archive, Serialize, Deserialize, Debug, Copy, Clone)]
pub enum XaeroAILayerSection {
    Backbone,
    Neck,
    Head,
    Unknown,
}

#[repr(C, align(64))]
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct XaeroAIModelLayer {
    pub layer_id: [u8; 32],
    pub layer_name: String,
    pub layer_type: XaeroLayerType,
    pub lora_target: bool,
    pub shape: [usize; 4],
    pub weights: Vec<u8>, // Raw quantized bytes
    pub quantization_info: QuantizationInfo,
    pub section: XaeroAILayerSection,
}

#[repr(C, align(64))]
#[derive(Archive, Serialize, Deserialize, Debug, Copy, Clone)]
pub struct QuantizationInfo {
    pub dtype: QuantizedDType,
    pub scale: f32,
    pub zero_point: i32,
}

#[repr(C, align(64))]
#[derive(Archive, Serialize, Deserialize, Debug, Copy, Clone)]
pub enum QuantizedDType {
    Q4_0, // 4-bit
    Q8_0, // 8-bit
    F16,  // Half precision
    F32,  // Full precision
}

pub struct XaeroModelArchitecture {
    pub backbone_layers: Vec<XaeroAIModelLayer>,
    pub neck_layers: Vec<XaeroAIModelLayer>,
    pub head_layers: Vec<XaeroAIModelLayer>,
}

#[repr(C)]
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Default)]
#[rkyv(derive(Debug))]
pub struct XaeroLoRALayerWeights {
    pub layer_id: [u8; 32],
    pub original_shape: [usize; 4], // [out_dim, in_dim] for the original layer
    pub lora_a: Vec<f32>,           // Flattened A matrix [in_dim, rank]
    pub lora_b: Vec<f32>,           // Flattened B matrix [rank, out_dim]
    pub scaling_factor: f64,        // alpha / rank
}

#[repr(C, align(64))]
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Default)]
#[rkyv(derive(Debug))]
pub struct XaeroLoRAAdapter {
    pub adapter_id: [u8; 32],
    pub base_model_hash: [u8; 32],
    pub user_id: [u8; 32],
    pub domain: String,
    pub rank: usize,
    pub alpha: f64,
    pub layer_adaptations: BTreeMap<[u8; 32], XaeroLoRALayerWeights>,
    pub training_metadata: LoRATrainingMeta,
}
#[repr(C)]
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Default)]
#[rkyv(derive(Debug))]
pub struct LoRATrainingMeta {
    pub epochs_trained: u32,
    pub final_loss: f64,
    pub dataset_hash: [u8; 32],
    pub training_time_ms: u64,
}

#[repr(C, align(64))]
#[derive(Archive, Serialize, Copy, Clone, Debug, Default)]
pub struct LoRAMetrics {
    // Performance metrics
    pub task_accuracy: f64,       // How well it performs on validation set
    pub base_model_accuracy: f64, // Original model performance for comparison
    pub improvement_percent: f64, // (lora_acc - base_acc) / base_acc

    // Training efficiency
    pub convergence_epoch: u32, // When loss stopped improving
    pub training_time_ms: u64,  // Total training duration
    pub final_loss: f64,
    pub best_loss: f64,

    // Resource usage
    pub parameter_efficiency: f64, // improvement_percent / (lora_params / total_params)
    pub memory_overhead_mb: f64,   // Extra memory needed for A,B matrices

    // Domain-specific (whiteboard)
    pub text_detection_f1: f64,    // How well it detects text regions
    pub diagram_detection_f1: f64, // How well it detects diagrams
    pub arrow_detection_f1: f64,   // How well it detects arrows/connectors

    // Stability metrics
    pub loss_variance: f64,         // How stable was training
    pub overfitting_indicator: f64, // validation_loss - training_loss
}

unsafe impl Pod for LoRAMetrics {}
unsafe impl Zeroable for LoRAMetrics {}

#[repr(C, align(64))]
#[derive(Debug, Clone, Archive, Serialize, Copy, Default)]
pub struct LoRAAdapterDiscovered {
    adapter_id: [u8; 32],
    base_model_hash: [u8; 32],
    domain: [u8; 32],
    performance_metrics: LoRAMetrics,
}
unsafe impl Pod for LoRAAdapterDiscovered {}
unsafe impl Zeroable for LoRAAdapterDiscovered {}
// Training Lifecycle
#[repr(C, align(64))]
#[derive(Debug, Clone, Archive, Serialize, Deserialize, Default)]
#[rkyv(derive(Debug))]
pub struct LoRATrainingRequested {
    base_model_id: [u8; 32],
    user_id: [u8; 32],
    target_layers: Vec<[u8; 32]>,
    hyperparams: LoRAHyperparams,
    dataset_id: [u8; 32],
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Archive, Serialize, Deserialize, Default)]
#[rkyv(derive(Debug))]
pub struct LoRAEpochCompleted {
    adapter_id: [u8; 32],
    epoch: u32,
    loss: f64,
    layer_updates: BTreeMap<[u8; 32], (Vec<f32>, Vec<f32>)>, // Updated A, B
}

// Composition & Inference
#[repr(C, align(64))]
#[derive(Debug, Clone, Archive, Serialize, Deserialize, Default)]
#[rkyv(derive(Debug))]
pub struct LoRACompositionRequested {
    base_model_id: [u8; 32],
    adapter_ids: Vec<[u8; 32]>, // Can stack multiple LoRAs
    inference_context: String,
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Archive, Serialize, Deserialize, Default)]
#[rkyv(derive(Debug))]
pub struct LoRAWeightsComposed {
    composition_id: [u8; 32],
    layer_deltas: BTreeMap<[u8; 32], Vec<f32>>, // Final composed weight deltas
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Archive, Serialize, Deserialize, Default)]
#[rkyv(derive(Debug))]
pub struct LoRAHyperparams {
    pub rank: usize,
    pub alpha: f64,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub max_epochs: u32,
    pub target_loss: f64,
}


pub fn classify_section(layer_name: &str) -> XaeroAILayerSection {
    if is_head_layer(layer_name) {
        XaeroAILayerSection::Head
    } else if is_neck_layer(layer_name) {
        XaeroAILayerSection::Neck
    } else {
        XaeroAILayerSection::Backbone
    }
}

pub fn is_head_layer(name: &str) -> bool {
    // YOLO head patterns
    name.contains("cv2") ||    // Classification head
        name.contains("cv3") ||    // Regression head
        name.contains("dfl") ||    // Distribution Focal Loss
        name.contains("detect") || // General detection head
        // Model index approach for YOLO
        (name.starts_with("model.") &&
            extract_model_index(name).map_or(false, |i| i >= 22))
}

pub fn is_neck_layer(name: &str) -> bool {
    name.contains("neck") ||
        name.contains("fpn") ||
        name.contains("upsample") ||
        // YOLO neck range (rough estimate)
        (name.starts_with("model.") &&
            extract_model_index(name).map_or(false, |i| i >= 10 && i < 22))
}

pub fn extract_model_index(name: &str) -> Option<u32> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() >= 2 && parts[0] == "model" {
        parts[1].parse().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
