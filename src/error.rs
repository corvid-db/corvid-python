//! Error mapping: engine errors → `CorvidErr` carrying the FFI error
//! code (docs/FFI.md §1.3, frozen 1..=19) so every failure surfaces as
//! a native `CorvidError` exception with `code` + `message`.
//!
//! pyo3's `#[pyclass(extends=PyException)]` route to attribute-carrying
//! exceptions requires `abi3-py312` (subclassing native types needs the
//! 3.12+ limited API); this binding targets 3.11+ with one cp311-abi3
//! wheel, so `CorvidError` is created with `create_exception!` and the
//! `code`/`message` attributes are set on the raised instance — CPython
//! exceptions carry a per-instance `__dict__`, so `e.code` behaves
//! identically for consumers. (The napi binding, by contrast, smuggles
//! the code through the thrown Error's message as JSON because napi-rs
//! cannot attach properties at all.)

use corvid::Error;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    corvid,
    CorvidError,
    PyException,
    "corvid engine error: carries the C-ABI error `code` (see ErrorCode) and the engine `message`."
);

/// The error-code table, value-identical to the C ABI's `corvid_err`
/// (docs/FFI.md §1.3). The golden fixtures pin behaviors to these
/// numbers, so the mapping must not drift from the engine enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum ErrCode {
    Database = 1,
    Transaction = 2,
    Table = 3,
    Storage = 4,
    Commit = 5,
    SetDurability = 6,
    Compaction = 7,
    Decode = 8,
    CorruptIndex = 9,
    ReservedCollection = 10,
    InvalidName = 11,
    Argument = 12,
    IncompatibleFormat = 13,
    EmptyIndexTraining = 14,
    SchemaViolation = 15,
    InvalidDump = 16,
    BackupTargetExists = 17,
    Io = 18,
    /// FFI/binding-only: compact while derived handles are open.
    Busy = 19,
}

impl ErrCode {
    pub fn num(self) -> u32 {
        self as u32
    }
}

/// The internal error type: an FFI code + engine message.
#[derive(Debug)]
pub struct CorvidErr {
    pub code: u32,
    pub message: String,
}

impl CorvidErr {
    pub fn new(code: ErrCode, message: impl Into<String>) -> Self {
        Self {
            code: code.num(),
            message: message.into(),
        }
    }

    pub fn argument(message: impl Into<String>) -> Self {
        Self::new(ErrCode::Argument, message)
    }
}

impl From<Error> for CorvidErr {
    fn from(e: Error) -> Self {
        let (code, message) = match &e {
            Error::Database(_) => (ErrCode::Database, e.to_string()),
            Error::Transaction(_) => (ErrCode::Transaction, e.to_string()),
            Error::Table(_) => (ErrCode::Table, e.to_string()),
            Error::Storage(_) => (ErrCode::Storage, e.to_string()),
            Error::Commit(_) => (ErrCode::Commit, e.to_string()),
            Error::SetDurability(_) => (ErrCode::SetDurability, e.to_string()),
            Error::Compaction(_) => (ErrCode::Compaction, e.to_string()),
            Error::Decode(_) => (ErrCode::Decode, e.to_string()),
            Error::CorruptIndex { context } => (
                ErrCode::CorruptIndex,
                format!("corrupt index state: {context}"),
            ),
            Error::ReservedCollection(n) => (
                ErrCode::ReservedCollection,
                format!("reserved collection name: {n}"),
            ),
            Error::InvalidName(n) => (
                ErrCode::InvalidName,
                format!("invalid name (NUL byte or `__` is not allowed): {n}"),
            ),
            Error::InvalidArgument(m) => (ErrCode::Argument, format!("invalid argument: {m}")),
            Error::IncompatibleFormat { found, expected } => (
                ErrCode::IncompatibleFormat,
                format!("incompatible format: file is v{found}, engine expects v{expected}"),
            ),
            Error::EmptyIndexTraining => (
                ErrCode::EmptyIndexTraining,
                "cannot train a PQ codebook: no usable training vectors".to_string(),
            ),
            Error::SchemaViolation(m) => {
                (ErrCode::SchemaViolation, format!("schema violation: {m}"))
            }
            Error::InvalidDump(m) => (ErrCode::InvalidDump, format!("invalid dump: {m}")),
            Error::BackupTargetExists(p) => (
                ErrCode::BackupTargetExists,
                format!("backup target already exists: {p}"),
            ),
            Error::Io(_) => (ErrCode::Io, e.to_string()),
            // The engine enum is #[non_exhaustive]; unknown future
            // variants surface as a storage-flavored error rather than
            // failing to compile the binding.
            _ => (ErrCode::Storage, e.to_string()),
        };
        CorvidErr {
            code: code.num(),
            message,
        }
    }
}

impl From<CorvidErr> for PyErr {
    fn from(e: CorvidErr) -> Self {
        // Force the exception instance and attach `code`/`message` as
        // instance attributes (str(e) renders the message via args).
        // Attach runs on a Python thread (every `?` crossing happens
        // inside a pymethod call), so the GIL is already held and
        // `Python::attach` is a cheap re-attach.
        Python::attach(|py| {
            let err = CorvidError::new_err(e.message.clone());
            let value = err.value(py);
            let _ = value.setattr("code", e.code);
            let _ = value.setattr("message", e.message);
            err
        })
    }
}

/// The frozen error-code table as class attributes
/// (`ErrorCode.INVALID_ARGUMENT`, `ErrorCode.BUSY`, ...) — the C ABI's
/// `corvid_err` numbering, never renumbered (docs/FFI.md §1.3).
#[pyclass]
pub struct ErrorCode {}

#[pymethods]
impl ErrorCode {
    #[classattr]
    const DATABASE: u32 = 1;
    #[classattr]
    const TRANSACTION: u32 = 2;
    #[classattr]
    const TABLE: u32 = 3;
    #[classattr]
    const STORAGE: u32 = 4;
    #[classattr]
    const COMMIT: u32 = 5;
    #[classattr]
    const SET_DURABILITY: u32 = 6;
    #[classattr]
    const COMPACTION: u32 = 7;
    #[classattr]
    const DECODE: u32 = 8;
    #[classattr]
    const CORRUPT_INDEX: u32 = 9;
    #[classattr]
    const RESERVED_COLLECTION: u32 = 10;
    #[classattr]
    const INVALID_NAME: u32 = 11;
    #[classattr]
    const INVALID_ARGUMENT: u32 = 12;
    #[classattr]
    const INCOMPATIBLE_FORMAT: u32 = 13;
    #[classattr]
    const EMPTY_INDEX_TRAINING: u32 = 14;
    #[classattr]
    const SCHEMA_VIOLATION: u32 = 15;
    #[classattr]
    const INVALID_DUMP: u32 = 16;
    #[classattr]
    const BACKUP_TARGET_EXISTS: u32 = 17;
    #[classattr]
    const IO: u32 = 18;
    #[classattr]
    const BUSY: u32 = 19;
}

pub type CResult<T> = Result<T, CorvidErr>;
