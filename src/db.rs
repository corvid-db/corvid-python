//! The `Db` pyclass. Holds the engine `Arc<Db>` plus the derived-handle
//! counter that gates exclusive compaction (the FFI's §4.13 rule,
//! mirrored: `compact` needs the counter at exactly 1 — the db itself —
//! AND sole `Arc` ownership, else code 19 `Busy`).

use std::fs::File;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use corvid::Db;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use crate::error::{CResult, CorvidErr, ErrCode};

/// The db's derived-handle counter: 1 for the db itself, +1 per live
/// `Collection`/`Query` (decremented by their `close`, their consuming
/// terminal op, or drop). Mirrors the FFI's `Arc<AtomicUsize>` so
/// `compact` keeps the same quiescence contract.
pub(crate) type Counter = Arc<AtomicUsize>;

pub(crate) struct DbInner {
    pub db: Arc<Db>,
    pub counter: Counter,
}

#[pyclass(name = "Db")]
pub struct DbPy {
    inner: Mutex<Option<DbInner>>,
}

pub(crate) fn retain(counter: &Counter) {
    counter.fetch_add(1, Ordering::SeqCst);
}

pub(crate) fn release(counter: &Counter) {
    counter.fetch_sub(1, Ordering::SeqCst);
}

pub(crate) fn closed_db_err() -> CorvidErr {
    CorvidErr::new(ErrCode::Argument, "database handle is closed")
}

impl DbPy {
    pub(crate) fn open_impl(path: Option<&str>) -> CResult<Self> {
        let db = match path {
            Some(p) => Db::open(p).map_err(CorvidErr::from)?,
            None => Db::open_in_memory().map_err(CorvidErr::from)?,
        };
        Ok(Self {
            inner: Mutex::new(Some(DbInner {
                db: Arc::new(db),
                counter: Arc::new(AtomicUsize::new(1)),
            })),
        })
    }

    pub(crate) fn with_inner<T>(&self, f: impl FnOnce(&DbInner) -> CResult<T>) -> CResult<T> {
        let guard = self.inner.lock().expect("db lock");
        match guard.as_ref() {
            Some(inner) => f(inner),
            None => Err(closed_db_err()),
        }
    }
}

#[pymethods]
impl DbPy {
    /// Open (or create) a database: `Db('app.redb')` for a file, `Db()`
    /// for a private in-memory one. (Prefer the named factories
    /// `Db.open` / `Db.open_memory` at call sites.)
    #[new]
    #[pyo3(signature = (path = None))]
    fn new(path: Option<&str>) -> PyResult<Self> {
        Ok(Self::open_impl(path)?)
    }

    /// Open (or create) a file-backed database at `path`.
    #[classmethod]
    fn open(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        Ok(Self::open_impl(Some(path))?)
    }

    /// Open a private, in-memory database.
    #[classmethod]
    fn open_memory(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        Ok(Self::open_impl(None)?)
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(&self, _py: Python<'_>, _exc: Option<Bound<'_, PyAny>>) {
        self.close();
    }

    fn __repr__(&self) -> String {
        match self.inner.lock().expect("db lock").as_ref() {
            Some(_) => "corvid.Db(<open>)".to_string(),
            None => "corvid.Db(<closed>)".to_string(),
        }
    }

    /// Acquire a collection handle (lazily created by the engine on
    /// first write; names are validated at write time, like the ABI).
    fn collection(&self, name: &str) -> PyResult<crate::CollectionPy> {
        Ok(self.with_inner(|inner| {
            retain(&inner.counter);
            Ok(crate::CollectionPy::new(
                Arc::clone(&inner.db),
                name.to_string(),
                Arc::clone(&inner.counter),
            ))
        })?)
    }

    /// The names of the database's collections, in engine order.
    fn collections(&self) -> PyResult<Vec<String>> {
        Ok(self.with_inner(|inner| inner.db.collections().map_err(CorvidErr::from))?)
    }

    /// Copy the database to `path` (which must not already exist).
    fn backup(&self, path: &str) -> PyResult<()> {
        Ok(self.with_inner(|inner| inner.db.backup(path).map_err(CorvidErr::from))?)
    }

    /// Dump the whole database (documents, indexes, schemas, TTLs,
    /// edges, auto-id counters) to `path`.
    fn dump_to_path(&self, path: &str) -> PyResult<()> {
        Ok(self.with_inner(|inner| {
            let file = File::create(path).map_err(io_err)?;
            inner.db.dump(file).map_err(CorvidErr::from)
        })?)
    }

    /// Replay a dump file into this database (merging).
    fn load_from_path(&self, path: &str) -> PyResult<()> {
        Ok(self.with_inner(|inner| {
            let file = File::open(path).map_err(io_err)?;
            inner.db.load(file).map_err(CorvidErr::from)
        })?)
    }

    /// Replay a dump file, renaming collections per `renames`
    /// (a `{from: to}` dict; targets validated before the stream is
    /// read).
    fn load_from_path_with_renames(&self, path: &str, renames: &Bound<'_, PyDict>) -> PyResult<()> {
        Ok(self.with_inner(|inner| {
            let file = File::open(path).map_err(io_err)?;
            let mut map = std::collections::BTreeMap::new();
            for (k, v) in renames.iter() {
                let from: String = k
                    .extract()
                    .map_err(|_| CorvidErr::argument("renames keys must be strings"))?;
                let to: String = v
                    .extract()
                    .map_err(|_| CorvidErr::argument("renames values must be strings"))?;
                map.insert(from, to);
            }
            inner
                .db
                .load_with_renames(file, &map)
                .map_err(CorvidErr::from)
        })?)
    }

    /// Compact the database file. Requires quiescence: every
    /// `Collection`/`Query` derived from this db must be closed (or
    /// have executed), otherwise a `Busy` CorvidError is raised.
    /// Returns whether any data was moved out.
    fn compact(&self) -> PyResult<bool> {
        Ok(self.compact_inner()?)
    }

    /// Close the handle (idempotent). Derived handles may legitimately
    /// outlive it — the engine lives until the last handle drops.
    fn close(&self) {
        let _ = self.inner.lock().expect("db lock").take();
    }
}

impl DbPy {
    fn compact_inner(&self) -> CResult<bool> {
        let mut guard = self.inner.lock().expect("db lock");
        let inner = guard.as_mut().ok_or_else(closed_db_err)?;
        if inner.counter.load(Ordering::SeqCst) != 1 {
            return Err(CorvidErr::new(
                ErrCode::Busy,
                "compact: derived handles are still open",
            ));
        }
        // Take the Arc out so exclusivity is observable, compact the
        // sole Db, re-share. `try_unwrap` failing means a handle raced
        // us — also Busy. While the lock is held the placeholder is
        // unobservable to every other call.
        let arc = std::mem::replace(&mut inner.db, Arc::new(placeholder_db()));
        match Arc::try_unwrap(arc) {
            Ok(mut db) => {
                let moved = match db.compact() {
                    Ok(m) => m,
                    Err(e) => {
                        inner.db = Arc::new(db);
                        return Err(CorvidErr::from(e));
                    }
                };
                inner.db = Arc::new(db);
                Ok(moved)
            }
            Err(arc) => {
                inner.db = arc;
                Err(CorvidErr::new(
                    ErrCode::Busy,
                    "compact: engine handles are still open",
                ))
            }
        }
    }
}

/// A stand-in while the real `Arc<Db>` is being unwrapped for
/// exclusive compaction (never observed: the mutex is held throughout).
fn placeholder_db() -> Db {
    Db::open_in_memory().expect("in-memory engine placeholder")
}

fn io_err(e: std::io::Error) -> CorvidErr {
    CorvidErr::from(corvid::Error::Io(e))
}
