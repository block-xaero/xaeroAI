use candle_core::quantized::QTensor;
use std::collections::BTreeMap;

pub struct TinyYolo {
    backbone: QTensor,
    neck: QTensor,
    head: QTensor,
    lora_adapters: BTreeMap<String, LoraLayer>,
}

#[repr(C, align(64))]
pub struct LoraLayer {
    a_weights: QTensor, // Low rank A matrix
    b_weights: QTensor, // Low rank B matrix
    rank: u8,
    alpha: f32,
}
