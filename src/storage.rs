use crate::{ArchivedXaeroLoRAAdapter, XaeroLoRAAdapter};
use lmdb::{Database, DatabaseFlags, Environment, Transaction, WriteFlags};
use rkyv::rancor::Failure;
use rkyv::{Archive, Deserialize, Serialize, from_bytes, rancor::Error, to_bytes};
use std::ffi::CString;
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};
use xaeroflux::actors::XaeroFlux;
use xaeroflux::actors::aof::storage::lmdb::{LmdbEnv, open_named_db};
use xaeroflux::hash::blake_hash_slice;
use xaeroid::XaeroID;

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

pub struct LmdbStore {
    pub env: Environment,
    pub lora_adapter_db: Database,
}
impl LmdbStore {
    pub fn create_env(&mut self) {
        unsafe {
            let res = lmdb::Environment::new().open(Path::new("xaeroai_store"));
            match res {
                Ok(env) => self.env = env,
                Err(e) => panic!("Error creating xaeroai_store environment: {e:?}"),
            }
        }
    }

    pub fn create_lora_adapter_db(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let rw = self.env.begin_rw_txn()?;
            let db = rw.create_db(Some("lora_adapter_db"), DatabaseFlags::default())?;
            tracing::debug!("db created: {db:?}");
            rw.commit()?;
            self.lora_adapter_db = db;
        }
        Ok(())
    }
    pub fn push_lora_adapter(
        &mut self,
        adapter: &XaeroLoRAAdapter,
    ) -> Result<[u8; 32], Box<dyn std::error::Error>> {
        unsafe {
            let bytes = rkyv::api::high::to_bytes(adapter)?.to_vec();
            let hash = blake_hash_slice(&bytes);
            tracing::debug!("hashed to {hash:?}");
            let mut tx = self.env.begin_rw_txn()?;
            tx.put::<[u8; 32], Vec<u8>>(
                self.lora_adapter_db,
                &hash,
                &bytes,
                WriteFlags::NO_DUP_DATA,
            )?;
            tx.commit()?;
            Ok(hash)
        }
    }

    pub fn get_lora_adapter_by_hash(
        &self,
        hash: [u8; 32],
    ) -> Result<&ArchivedXaeroLoRAAdapter, Box<dyn std::error::Error>> {
        let mut tx = self.env.begin_ro_txn()?;
        let res = tx.get(self.lora_adapter_db, &hash)?;
        let res = rkyv::api::high::access::<ArchivedXaeroLoRAAdapter, Failure>(res)?;
        tx.commit()?;
        Ok(res)
    }
}
