use crate::LoRATrainingRequested;
use rusted_ring::PooledEvent;
use std::sync::OnceLock;
use rkyv::rancor::Failure;
use xaeroflux::actors::{XaeroFlux, XaeroFluxError};
use xaeroflux::event::EventType;
use xaeroid::XaeroID;
/*
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
*/

static HANDLE: OnceLock<XaeroFlux> = OnceLock::new();

pub fn get_xaeroflux_handle(xid: XaeroID) -> &'static XaeroFlux {
    let mut xf = HANDLE.get_or_init(|| {
        let mut xf = XaeroFlux::new();
        xf.start_aof().expect("AOF failed to initialize");
        xf.start_p2p(xid).expect("P2P failed to initialize");
        xf
    });
    xf
}
