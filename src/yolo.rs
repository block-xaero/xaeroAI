use crate::storage::LmdbStore;
use crate::{XaeroAILayerSection, XaeroAIModel, XaeroAIModelOps, XaeroLayerType};
use candle_core::{Device, Tensor};
use std::error::Error;
use std::sync::{Arc, Mutex};

pub struct YoloLoraAdaptedModel {
    adapter_db: Arc<Mutex<LmdbStore>>,
    pub xaero_aimodel: XaeroAIModel,
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
            if ((parts[2].contains("head") ||
               parts[2].contains("classifier") ||
               parts[2].contains("fc") ||
               parts[2].contains("detect") ||
               parts[2].contains("dfl"))
              && model_idx > 22) {
            }
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
