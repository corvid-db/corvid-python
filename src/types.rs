//! Small result/dataclasses of the public API: `SchemaField`, `Row`,
//! `Page`, `GeoHit`. All attribute-exposed (`row.key`), immutable from
//! Python, and fully typed in `python/corvid/__init__.pyi`.

use pyo3::prelude::*;
use pyo3::types::PyAny;

/// One field of a declared schema (`Collection.set_schema` /
/// `Collection.schema`). `ty` is one of
/// `any|bool|int|float|text|bytes|vector|array|map`.
#[pyclass(name = "SchemaField", get_all, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct SchemaField {
    /// The field's name.
    pub name: String,
    /// The field's type (`any|bool|int|float|text|bytes|vector|array|map`).
    pub ty: String,
    /// Whether the field must be present (not None).
    pub required: bool,
    /// Whether values must be unique across the collection.
    pub unique: bool,
}

#[pymethods]
impl SchemaField {
    #[new]
    #[pyo3(signature = (name, ty, required = false, unique = false))]
    fn new(name: String, ty: String, required: bool, unique: bool) -> Self {
        Self {
            name,
            ty,
            required,
            unique,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SchemaField(name={:?}, ty={:?}, required={}, unique={})",
            self.name, self.ty, self.required, self.unique
        )
    }
}

/// One query result row (`Query.run`).
#[pyclass(name = "Row", get_all)]
pub struct Row {
    /// The document's key (`str`, or `bytes` for non-UTF-8 keys).
    pub key: Py<PyAny>,
    /// The rank score (RRF-fused, or `0.0` for pure filter/order queries).
    pub score: f32,
    /// The full (or projected) stored document.
    pub document: Py<PyAny>,
}

/// One page of keyset-paginated rows (`Collection.page`).
#[pyclass(name = "Page", get_all)]
pub struct Page {
    /// The `(key, doc)` rows, in key order.
    pub rows: Vec<(Py<PyAny>, Py<PyAny>)>,
    /// The resume cursor (pass as the next call's `after`), or `None`
    /// at the end.
    pub next: Option<Py<PyAny>>,
}

/// One geo-search hit (`Collection.geo_*`).
#[pyclass(name = "GeoHit", get_all)]
pub struct GeoHit {
    /// The document's key.
    pub key: Py<PyAny>,
    /// Distance from the query center in kilometres (the 0.0 sentinel
    /// for bbox searches, which have no center).
    pub distance_km: f64,
    /// The full stored document.
    pub document: Py<PyAny>,
}
