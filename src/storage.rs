use crate::{ArchivedXaeroLoRAAdapter, XaeroLoRAAdapter};
use liblmdb::{
    mdb_env_create, mdb_put, mdb_txn_abort, mdb_txn_begin, mdb_txn_commit, MDB_dbi, MDB_env, MDB_txn,
    MDB_val, MDB_APPEND, MDB_CREATE, MDB_RDONLY,
};
use rkyv::rancor::Failure;
use std::sync::OnceLock;
use xaeroflux::actors::aof::storage::lmdb::{from_lmdb_err, open_named_db};
use xaeroflux::actors::XaeroFlux;
use xaeroflux::hash::blake_hash_slice;
use xaeroid::XaeroID;

static HANDLE: OnceLock<XaeroFlux> = OnceLock::new();

pub fn get_xaeroflux_handle(xid: XaeroID) -> &'static XaeroFlux {
    (HANDLE.get_or_init(|| {
        let mut xf = XaeroFlux::new();
        xf.start_aof().expect("AOF failed to initialize");
        xf.start_p2p(xid).expect("P2P failed to initialize");
        xf
    })) as _
}

pub struct LmdbStore {
    pub env: *mut MDB_env,
    pub lora_adapter_db: MDB_dbi,
}
impl LmdbStore {
    pub fn new() -> Self {
        let env = std::ptr::null_mut();
        unsafe {
            let env_creation_result = mdb_env_create(env as *mut _);
            if env_creation_result != 0 {
                panic!("mdb_env_create failed with {env_creation_result:?}");
            }
        };
        let lora_adapter_db: MDB_dbi = unsafe {
            match open_named_db(env, c"/lora_adapters_db".as_ptr()) {
                Ok(dbi) => dbi,
                Err(e) => {
                    panic!("open_named_db failed with {e:?}");
                }
            }
        };
        Self {
            env,
            lora_adapter_db,
        }
    }

    pub fn push_lora_adapter_db(
        &mut self,
        adapter: XaeroLoRAAdapter,
    ) -> Result<[u8; 32], Box<dyn std::error::Error>> {
        let tx = self.tx_begin();
        let bytes = rkyv::to_bytes::<Failure>(&adapter)?;
        let hash = blake_hash_slice(&bytes);
        let mut mdb_key = MDB_val {
            mv_size: 32,
            mv_data: hash.as_ptr() as *mut _,
        };
        let mut mdb_value = MDB_val {
            mv_size: bytes.len(),
            mv_data: bytes.as_ptr() as *mut _,
        };
        unsafe {
            mdb_put(
                tx,
                self.lora_adapter_db,
                &mut mdb_key,
                &mut mdb_value,
                MDB_APPEND,
            )
        };
        self.tx_commit(tx);
        Ok(hash)
    }

    pub fn get_lora_adapter_db_by_hash(
        &self,
        adapter_key: &[u8; 32],
    ) -> Result<Option<&ArchivedXaeroLoRAAdapter>, Box<dyn std::error::Error>> {
        unsafe {
            let mut txn: *mut MDB_txn = std::ptr::null_mut();
            let rc = mdb_txn_begin(self.env, std::ptr::null_mut(), MDB_RDONLY, &mut txn);
            if rc != 0 {
                return Err(from_lmdb_err(rc));
            }

            let mut key_val = MDB_val {
                mv_size: adapter_key.len(),
                mv_data: adapter_key.as_ptr() as *mut _,
            };
            let mut data_val = MDB_val {
                mv_size: 0,
                mv_data: std::ptr::null_mut(),
            };

            let getrc = liblmdb::mdb_get(txn, self.lora_adapter_db, &mut key_val, &mut data_val);
            if getrc != 0 {
                mdb_txn_abort(txn);
                if getrc == liblmdb::MDB_NOTFOUND {
                    return Ok(None);
                } else {
                    return Err(from_lmdb_err(getrc));
                }
            }

            let slice = std::slice::from_raw_parts(data_val.mv_data as *const u8, data_val.mv_size);
            let event_key_found = rkyv::access::<ArchivedXaeroLoRAAdapter, Failure>(slice)?;
            Ok(Some(event_key_found))
        }
    }
    pub fn tx_begin(&mut self) -> *mut MDB_txn {
        unsafe {
            let mut txn: *mut MDB_txn = std::ptr::null_mut();
            let return_code = mdb_txn_begin(self.env, std::ptr::null_mut(), MDB_CREATE, &mut txn);
            if return_code != 0 {
                panic!("tx_begin failed with {return_code:?}");
            }
            txn
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn tx_commit(&mut self, txn: *mut MDB_txn) -> *mut MDB_txn {
        let return_code = unsafe { mdb_txn_commit(txn) };
        if return_code != 0 {
            panic!("tx_commit failed with {return_code:?}");
        }
        txn
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn tx_abort(&mut self, txn: *mut MDB_txn) {
        unsafe { mdb_txn_abort(txn) }
    }
}
