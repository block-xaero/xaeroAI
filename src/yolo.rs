use crate::storage::LmdbStore;
use crate::{classify_section, QuantizationInfo, QuantizedDType, XaeroAILayerSection, XaeroAIModel, XaeroAIModelLayer, XaeroAIModelOps, XaeroLayerType};
use candle_core::{Device, Tensor};
use std::error::Error;
use std::sync::{Arc, Mutex};

pub struct YoloLoraAdaptedModel {
    adapter_db: Arc<Mutex<LmdbStore>>,
    pub xaero_aimodel: XaeroAIModel,
}

impl YoloLoraAdaptedModel {
    fn infer_type_from_name(name: &str) -> XaeroLayerType {
        if name.contains("conv") { XaeroLayerType::Conv } else if name.contains("linear") || name.contains("fc") { XaeroLayerType::Linear } else { XaeroLayerType::Conv } // default
    }

    fn should_adapt_layer(name: &str) -> bool {
        name.contains("cv2") || name.contains("cv3") || name.contains("dfl")
    }

    pub fn from_tensor_with_name(tensor: &Tensor, layer_name: &str) -> XaeroAIModelLayer {
        XaeroAIModelLayer {
            layer_id: blake3::hash(layer_name.as_bytes()).into(),
            layer_name: layer_name.to_string(),
            layer_type: Self::infer_type_from_name(layer_name),
            lora_target: Self::should_adapt_layer(layer_name),
            shape: tensor.shape().dims4().unwrap().into(),
            weights: vec![], // Empty for mmap
            quantization_info: QuantizationInfo {
                dtype: QuantizedDType::Q4_0,
                scale: 0.0,
                zero_point: 0,
            },
            section: classify_section(layer_name),
        }
    }
}
#[allow(unused_variables)]
impl XaeroAIModelOps for YoloLoraAdaptedModel {
    fn forward_with_lora(
        &self,
        input: &Tensor,
        user_id: [u8; 32],
    ) -> Result<Tensor, Box<dyn Error>> {
        let layer_to_tensors =
            candle_core::safetensors::load("models/yolo11n.safetensors", &Device::Cpu)?;
        for tensor_entry in layer_to_tensors.iter() {
            // model.X.cv1.conv.weight"
            let split_layer_name = tensor_entry.0.split(".");
            let vec_from_split_layer_name = split_layer_name.collect::<Vec<&str>>();
            let parts = vec_from_split_layer_name.as_slice();
            let model_idx = parts[1].parse::<u32>()?;
            let component = parts[2].parse::<u8>()?;
            let operation = parts[3].parse::<u8>()?;
            
        }

        // mmapd_safetensor.get()
        let lora_adapter_hash = self
            .xaero_aimodel
            .lora_adapters
            .get(&user_id)
            .expect("cannot find lora adapter hash for the user -- fail fast!");
        match self.adapter_db.lock() {
            Ok(mut adapter_db) => {
                let res = adapter_db.get_lora_adapter_db_by_hash(*lora_adapter_hash)?;
                match res {
                    None => {
                        panic!("cannot get lora adapter db for user -- fail fast!");
                    }
                    Some(adapter) => {}
                }
                Ok(input.clone())
            }
            Err(e) => {
                // todo: unsure if we fail fast or not
                panic!("failed to lock to get adapter {e:?}");
            }
        }
    }
}
