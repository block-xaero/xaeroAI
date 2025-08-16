use crate::storage::LmdbStore;
use crate::{XaeroAIModel, XaeroAIModelOps};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::error::Error;
use std::path::Path;
use std::sync::{Arc, Mutex};
use xaeroid::XaeroID;

pub struct YoloLoraAdaptedModel {
    adapter_db: Arc<Mutex<LmdbStore>>,
    pub xaero_aimodel: XaeroAIModel,
}
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
        let adapter_db = self.adapter_db.lock()?;
        let res = adapter_db.get_lora_adapter_by_hash(*lora_adapter_hash)?;
        Ok(Tensor::from(input.clone()))
    }
}
