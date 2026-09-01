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

#[cfg(test)]
mod variant_inventory {
    //! The FFI crate's variant-inventory discipline, ported (its
    //! src/error.rs tests): the engine enum is `#[non_exhaustive]`,
    //! and `From<Error> for CorvidErr` above ends in a wildcard arm
    //! that would silently map an UNKNOWN future variant to code 4.
    //! This snapshot test constructs one instance of every variant of
    //! the REAL enum (the engine is compiled in — redb's public error
    //! enums are the only constructors for the six passthrough
    //! variants) and matches it through a wildcard-armed name
    //! extractor, so:
    //!
    //! * removing/renaming an engine variant fails COMPILATION (the
    //!   named arms in `From` and in `assert_mapped` below);
    //! * a variant outside the pinned inventory trips the wildcard
    //!   here, loudly, instead of silently reading as code 4;
    //! * each known variant still maps to its frozen FFI code.
    //!
    //! An engine ADDITION is only caught once `every_engine_variant`
    //! is extended in step — the same hand-maintained-list gap the
    //! engine's `enum Error` doc comment points at (extend that list,
    //! the `From` impl, and FFI.md §1.3 together).
    use super::{CorvidErr, ErrCode};
    use corvid::Error;

    /// The pinned engine tag's variant set, in frozen code order
    /// 1..=18 (FFI.md §1.3 — the same table the C ABI pins).
    const ENGINE_VARIANT_INVENTORY: [&str; 18] = [
        "Database",
        "Transaction",
        "Table",
        "Storage",
        "Commit",
        "SetDurability",
        "Compaction",
        "Decode",
        "CorruptIndex",
        "ReservedCollection",
        "InvalidName",
        "InvalidArgument",
        "IncompatibleFormat",
        "EmptyIndexTraining",
        "SchemaViolation",
        "InvalidDump",
        "BackupTargetExists",
        "Io",
    ];

    /// The inventory in code order, as (name, ErrCode) pairs.
    const SPEC_MAPPING: [(&str, ErrCode); 18] = [
        ("Database", ErrCode::Database),
        ("Transaction", ErrCode::Transaction),
        ("Table", ErrCode::Table),
        ("Storage", ErrCode::Storage),
        ("Commit", ErrCode::Commit),
        ("SetDurability", ErrCode::SetDurability),
        ("Compaction", ErrCode::Compaction),
        ("Decode", ErrCode::Decode),
        ("CorruptIndex", ErrCode::CorruptIndex),
        ("ReservedCollection", ErrCode::ReservedCollection),
        ("InvalidName", ErrCode::InvalidName),
        ("InvalidArgument", ErrCode::Argument),
        ("IncompatibleFormat", ErrCode::IncompatibleFormat),
        ("EmptyIndexTraining", ErrCode::EmptyIndexTraining),
        ("SchemaViolation", ErrCode::SchemaViolation),
        ("InvalidDump", ErrCode::InvalidDump),
        ("BackupTargetExists", ErrCode::BackupTargetExists),
        ("Io", ErrCode::Io),
    ];

    /// Match every engine variant by name, with a wildcard arm that
    /// FAILS on anything outside the inventory — the drift detector.
    fn assert_mapped(err: &Error) -> &'static str {
        match err {
            Error::Database(_) => "Database",
            Error::Transaction(_) => "Transaction",
            Error::Table(_) => "Table",
            Error::Storage(_) => "Storage",
            Error::Commit(_) => "Commit",
            Error::SetDurability(_) => "SetDurability",
            Error::Compaction(_) => "Compaction",
            Error::Decode(_) => "Decode",
            Error::CorruptIndex { .. } => "CorruptIndex",
            Error::ReservedCollection(_) => "ReservedCollection",
            Error::InvalidName(_) => "InvalidName",
            Error::InvalidArgument(_) => "InvalidArgument",
            Error::IncompatibleFormat { .. } => "IncompatibleFormat",
            Error::EmptyIndexTraining => "EmptyIndexTraining",
            Error::SchemaViolation(_) => "SchemaViolation",
            Error::InvalidDump(_) => "InvalidDump",
            Error::BackupTargetExists(_) => "BackupTargetExists",
            Error::Io(_) => "Io",
            unexpected => panic!(
                "corvid::Error has a variant outside ENGINE_VARIANT_INVENTORY \
                 ({unexpected:?}) — extend the inventory, the From<Error> mapping, \
                 and ErrCode before shipping (an unknown variant currently \
                 surfaces as code 4 via the wildcard)"
            ),
        }
    }

    /// One constructible instance of every KNOWN engine variant, built
    /// from redb's public error enums for the six passthrough variants
    /// (the engine wraps them via `#[from]`).
    fn every_engine_variant() -> Vec<Error> {
        use redb::{
            CommitError, CompactionError, DatabaseError, SetDurabilityError, StorageError,
            TableError, TransactionError,
        };
        vec![
            Error::Database(DatabaseError::DatabaseAlreadyOpen),
            Error::Transaction(TransactionError::Storage(StorageError::DatabaseClosed)),
            Error::Table(TableError::TableDoesNotExist("t".into())),
            Error::Storage(StorageError::DatabaseClosed),
            Error::Commit(CommitError::Storage(StorageError::DatabaseClosed)),
            Error::SetDurability(SetDurabilityError::PersistentSavepointModified),
            Error::Compaction(CompactionError::TransactionInProgress),
            Error::Decode("bad bytes".into()),
            Error::CorruptIndex {
                context: "truncated".into(),
            },
            Error::ReservedCollection("__x".into()),
            Error::InvalidName("a__b".into()),
            Error::InvalidArgument("lambda out of range".into()),
            Error::IncompatibleFormat {
                found: 1,
                expected: 2,
            },
            Error::EmptyIndexTraining,
            Error::SchemaViolation("field f".into()),
            Error::InvalidDump("unknown version".into()),
            Error::BackupTargetExists("/tmp/old".into()),
            Error::Io(std::io::Error::other("gone")),
        ]
    }

    /// The snapshot: the pinned engine tag's variant set maps onto the
    /// frozen code table exactly — inventory == ErrCode == engine, and
    /// nothing falls through the `From` wildcard.
    #[test]
    fn variant_inventory_matches_the_engine_and_the_mapping() {
        let variants = every_engine_variant();
        assert_eq!(
            variants.len(),
            ENGINE_VARIANT_INVENTORY.len(),
            "constructor list and inventory disagree"
        );
        assert_eq!(
            ENGINE_VARIANT_INVENTORY.len(),
            SPEC_MAPPING.len(),
            "inventory and spec mapping tables disagree"
        );
        for ((inventory_name, (spec_name, code)), err) in ENGINE_VARIANT_INVENTORY
            .iter()
            .zip(SPEC_MAPPING.iter())
            .zip(variants.into_iter())
        {
            assert_eq!(inventory_name, spec_name, "inventory/mapping order drift");
            let name = assert_mapped(&err); // wildcard trips on drift
            assert_eq!(name, *inventory_name);
            let mapped = CorvidErr::from(err);
            assert_eq!(
                mapped.code,
                code.num(),
                "mapping drift for {name}: {:?}",
                mapped.code
            );
        }
    }
}
