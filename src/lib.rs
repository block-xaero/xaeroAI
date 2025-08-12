use candle_core::quantized::QuantizedType;
use candle_core::{Shape, shape};
use std::collections::BTreeMap;
use xaeroid::XaeroID;

pub struct XaeroAIModelRegistry {
    pub xaero_id: XaeroID,
    pub models: Vec<XaeroAIModel>,
}

pub struct XaeroAIModel {
    pub xaero_id: XaeroID,
    pub model_hash: [u8; 32],
    pub arch: XaeroModelArchitecture,
    pub quantization: Box<dyn QuantizedType>, // INT8, INT4, F16, etc.
    pub lora_adapters: BTreeMap<XaeroID, XaeroID>, // user_id -> lora_delta_hash
    pub base_size_bytes: u64,                 // For P2P transfer planning
}

pub enum XaeroLayerType {
    Conv,
    Linear,
    Attention,
}
pub struct XaeroAIModelLayer {
    pub layer_id: [u8; 32],
    pub layer_type: XaeroLayerType,
    pub lora_target: bool,
    pub shape: shape::Shape,
}
pub struct XaeroModelArchitecture {
    pub backbone_layers: Vec<XaeroAIModelLayer>,
    pub neck_layers: Vec<XaeroAIModelLayer>,
    pub head_layers: Vec<XaeroAIModelLayer>,
}

pub struct XaeroLoRALayerWeights {
    pub layer_id: XaeroID,
    pub original_shape: Shape, // [out_dim, in_dim] for the original layer
    pub lora_a: Vec<f32>,      // Flattened A matrix [in_dim, rank]
    pub lora_b: Vec<f32>,      // Flattened B matrix [rank, out_dim]
    pub scaling_factor: f64,   // alpha / rank
}

pub struct XaeroLoRAAdapter {
    pub adapter_id: XaeroID,
    pub base_model_hash: [u8; 32],
    pub user_id: XaeroID,
    pub domain: String, // "whiteboard", "medical", "automotive"
    pub rank: usize,
    pub alpha: f64,
    pub layer_adaptations: BTreeMap<XaeroID, XaeroLoRALayerWeights>,
    pub training_metadata: LoRATrainingMeta,
}

pub struct LoRATrainingMeta {
    pub epochs_trained: u32,
    pub final_loss: f64,
    pub dataset_hash: [u8; 32],
    pub training_time_ms: u64,
}
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

pub enum LoRAEvent {
    // Discovery & Loading
    LoRAAdapterDiscovered {
        adapter_id: XaeroID,
        base_model_hash: [u8; 32],
        domain: String,
        performance_metrics: Option<LoRAMetrics>,
    },

    // Training Lifecycle
    LoRATrainingRequested {
        base_model_id: XaeroID,
        user_id: XaeroID,
        target_layers: Vec<XaeroID>,
        hyperparams: LoRAHyperparams,
        dataset_id: XaeroID,
    },

    LoRAEpochCompleted {
        adapter_id: XaeroID,
        epoch: u32,
        loss: f64,
        layer_updates: BTreeMap<XaeroID, (Vec<f32>, Vec<f32>)>, // Updated A, B
    },

    // Composition & Inference
    LoRACompositionRequested {
        base_model_id: XaeroID,
        adapter_ids: Vec<XaeroID>, // Can stack multiple LoRAs
        inference_context: String,
    },

    LoRAWeightsComposed {
        composition_id: XaeroID,
        layer_deltas: BTreeMap<XaeroID, Vec<f32>>, // Final composed weight deltas
    },
}

pub struct LoRAHyperparams {
    pub rank: usize,
    pub alpha: f64,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub max_epochs: u32,
    pub target_loss: f64,
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
