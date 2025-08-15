use std::error::Error;
use std::sync::{Arc, Mutex};
use candle_core::{Device, Tensor};
use xaeroid::XaeroID;
use crate::{XaeroAIModel, XaeroAIModelOps};
use crate::storage::LmdbStore;

pub struct YoloLoraAdaptedModel{
    adapter_db: Arc<Mutex<LmdbStore>>,
    pub xaero_aimodel: XaeroAIModel
}
impl XaeroAIModelOps for YoloLoraAdaptedModel{
    pub fn forward_with_lora(&self, input: &Tensor, user_id: [u8;32]) -> Result<Tensor, Box<dyn
    Error>> {
        let lora_adapter_hash = self.xaero_aimodel.lora_adapters.get(&user_id).expect("cannot \
        find lora adapter hash for the user -- fail fast!");
        let adapter_db = self.adapter_db.lock()?;
        let res = adapter_db.get_lora_adapter_by_hash(*lora_adapter_hash)?;

    }
}