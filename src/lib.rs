//! corvid-python — the Python binding for the corvid engine.
//!
//! This crate is the *engine-binding layer* AND the idiom layer: pyo3
//! supports native exception subclasses, properties, and fluent
//! chaining, so the public surface (`Db`, `Collection`, `Query`,
//! `field()`, `CorvidError`) is implemented directly in Rust and
//! exposed as the `corvid` package (the `corvid._native` module,
//! re-exported by `python/corvid/__init__.py`; types are declared in
//! `python/corvid/__init__.pyi`, with `py.typed`). See docs/PLAN.md
//! for the architecture ruling and §8 idiom gate compliance.

use pyo3::prelude::*;

mod collection;
mod db;
mod error;
mod pred;
mod query;
mod types;
mod value;

pub use collection::CollectionPy;
pub use db::DbPy;
pub use error::{CorvidError, ErrorCode};
pub use query::QueryPy;

/// The FFI-ABI generation this binding's OOP surface covers
/// (docs/FFI.md §1.3 stability policy; `corvid_ffi_version` = 1).
#[pyfunction]
pub fn ffi_version() -> u32 {
    1
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DbPy>()?;
    m.add_class::<CollectionPy>()?;
    m.add_class::<QueryPy>()?;
    // Exceptions are not pyclasses: register the created exception's
    // type object directly (pyo3's documented module pattern).
    m.add("CorvidError", m.py().get_type::<error::CorvidError>())?;
    m.add_class::<error::ErrorCode>()?;
    m.add_class::<pred::Pred>()?;
    m.add_class::<pred::FieldRefBuilder>()?;
    m.add_class::<types::SchemaField>()?;
    m.add_class::<types::Row>()?;
    m.add_class::<types::Page>()?;
    m.add_class::<types::GeoHit>()?;
    m.add_function(wrap_pyfunction!(ffi_version, m)?)?;
    m.add_function(wrap_pyfunction!(pred::field, m)?)?;
    m.add_function(wrap_pyfunction!(pred::and_, m)?)?;
    m.add_function(wrap_pyfunction!(pred::or_, m)?)?;
    m.add_function(wrap_pyfunction!(pred::not_, m)?)?;
    Ok(())
}
