use candle_core::quantized::QuantizedType;
use candle_core::shape;
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
#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
