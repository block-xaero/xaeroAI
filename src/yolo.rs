use crate::storage::LmdbStore;
use crate::{XaeroAIModel, XaeroAIModelOps};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::error::Error;
use std::path::Path;
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
        let mmapd_safetensor = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[Path::new("models/yolo11n.safetensors")],
                DType::F32,
                &Device::Cpu,
            )
        };
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
                    Some(adapter) => {
                        //
                    }
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
