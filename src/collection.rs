//! The `Collection` pyclass. Holds `Arc<Db>` + name (the ABI's derived
//! handle shape); each op materializes the engine `Collection` for the
//! call, mirroring the FFI's transient-borrow pattern.

use std::sync::{Arc, Mutex};

use corvid::schema::{Field, FieldType, Schema};
use corvid::{Collection, GeoHit as EngineGeoHit, Metric, Quantization};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::db::{release, Counter};
use crate::error::{CResult, CorvidErr, ErrCode};
use crate::pred::Pred;
use crate::types::{GeoHit, Page, SchemaField};
use crate::value::{key_from_py, key_to_py, out, value_from_py, value_to_py};

pub(crate) struct CollInner {
    pub db: Arc<corvid::Db>,
    pub name: String,
    pub counter: Counter,
}

#[pyclass(name = "Collection")]
pub struct CollectionPy {
    inner: Mutex<Option<CollInner>>,
}

pub(crate) fn parse_metric(s: &str) -> CResult<Metric> {
    match s {
        "cosine" => Ok(Metric::Cosine),
        "dot" => Ok(Metric::Dot),
        "l2" => Ok(Metric::L2),
        _ => Err(CorvidErr::new(
            ErrCode::Argument,
            format!("unknown metric '{s}' (want cosine|dot|l2)"),
        )),
    }
}

pub(crate) fn parse_quant(s: &str) -> CResult<Quantization> {
    match s {
        "none" => Ok(Quantization::None),
        "binary" => Ok(Quantization::Binary),
        "scalar" => Ok(Quantization::Scalar),
        _ => Err(CorvidErr::new(
            ErrCode::Argument,
            format!("unknown quantization '{s}' (want none|binary|scalar)"),
        )),
    }
}

pub(crate) fn parse_field_type(s: &str) -> CResult<FieldType> {
    match s {
        "any" => Ok(FieldType::Any),
        "bool" => Ok(FieldType::Bool),
        "int" => Ok(FieldType::Int),
        "float" => Ok(FieldType::Float),
        "text" => Ok(FieldType::Text),
        "bytes" => Ok(FieldType::Bytes),
        "vector" => Ok(FieldType::Vector),
        "array" => Ok(FieldType::Array),
        "map" => Ok(FieldType::Map),
        _ => Err(CorvidErr::new(
            ErrCode::Argument,
            format!(
                "unknown field type '{s}' (want any|bool|int|float|text|bytes|vector|array|map)"
            ),
        )),
    }
}

pub(crate) fn field_type_name(t: FieldType) -> &'static str {
    match t {
        FieldType::Any => "any",
        FieldType::Bool => "bool",
        FieldType::Int => "int",
        FieldType::Float => "float",
        FieldType::Text => "text",
        FieldType::Bytes => "bytes",
        FieldType::Vector => "vector",
        FieldType::Array => "array",
        FieldType::Map => "map",
    }
}

fn closed() -> CorvidErr {
    CorvidErr::new(ErrCode::Argument, "collection handle is closed")
}

impl CollectionPy {
    pub(crate) fn new(db: Arc<corvid::Db>, name: String, counter: Counter) -> Self {
        Self {
            inner: Mutex::new(Some(CollInner { db, name, counter })),
        }
    }

    fn with_coll<T>(&self, f: impl FnOnce(Collection<'_>) -> CResult<T>) -> CResult<T> {
        let guard = self.inner.lock().expect("collection lock");
        let inner = guard.as_ref().ok_or_else(closed)?;
        let coll = inner.db.collection(&inner.name);
        f(coll)
    }
}

impl Drop for CollectionPy {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.lock().expect("collection lock").take() {
            release(&inner.counter);
        }
    }
}

/// `update`/`scan_each` callbacks: Python-side failures (conversion or
/// a raising callback) crossed inside a closure that must return
/// `corvid::Result` ride in an InvalidArgument, exactly like the napi
/// binding.
fn engine_err(e: PyErr) -> corvid::Error {
    corvid::Error::InvalidArgument(e.to_string())
}

#[pymethods]
impl CollectionPy {
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(&self, _py: Python<'_>, _exc: Option<Bound<'_, PyAny>>) {
        self.close();
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.with_coll(|coll| coll.len().map_err(CorvidErr::from))?)
    }

    fn __repr__(&self) -> String {
        match self.inner.lock().expect("collection lock").as_ref() {
            Some(inner) => format!("corvid.Collection({:?})", inner.name),
            None => "corvid.Collection(<closed>)".to_string(),
        }
    }

    /// The collection's name.
    #[getter]
    fn name(&self) -> PyResult<String> {
        let guard = self.inner.lock().expect("collection lock");
        Ok(guard.as_ref().ok_or_else(closed)?.name.clone())
    }

    // -- mutations ----------------------------------------------------------

    /// Insert (replace) `doc` at `key`.
    fn insert(&self, key: &Bound<'_, PyAny>, doc: &Bound<'_, PyAny>) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            let k = key_from_py(key)?;
            let v = value_from_py(doc)?;
            coll.insert(&k, &v).map_err(CorvidErr::from)
        })?)
    }

    /// Bulk atomic insert (`put_many`): one transaction; a violating
    /// pair rolls the whole batch back.
    fn insert_many(&self, entries: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)>) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            let mut items: Vec<(Vec<u8>, corvid::Value)> = Vec::with_capacity(entries.len());
            for (k, v) in &entries {
                items.push((key_from_py(k)?, value_from_py(v)?));
            }
            let refs: Vec<(&[u8], &corvid::Value)> =
                items.iter().map(|(k, v)| (k.as_slice(), v)).collect();
            coll.insert_batch(&refs).map_err(CorvidErr::from)
        })?)
    }

    /// Insert with an engine-generated key (20-digit, strictly
    /// monotonic per collection); returns the key.
    fn insert_auto(&self, py: Python<'_>, doc: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let key = self.with_coll(|coll| {
            let v = value_from_py(doc)?;
            coll.insert_auto(&v).map_err(CorvidErr::from)
        })?;
        key_to_py(py, &key)
    }

    /// Read-modify-write: `fn` receives the current document (or `None`
    /// when absent) and returns the new document — `None` to delete. A
    /// raising callback aborts with code 12 and writes nothing. `fn`
    /// must NOT call methods on this same Collection: the handle's lock
    /// is non-reentrant (the FFI's portable contract), so a reentrant
    /// call deadlocks. (The engine's own `update` is the same
    /// get-then-write composition; see its docs for the linearizability
    /// caveat.)
    fn update(&self, py: Python<'_>, key: &Bound<'_, PyAny>, f: &Bound<'_, PyAny>) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            let k = key_from_py(key)?;
            let current = coll.get(&k).map_err(CorvidErr::from)?;
            let arg = match &current {
                Some(v) => value_to_py(py, v).map_err(engine_err)?,
                None => py.None(),
            };
            let ret = f
                .call1((arg,))
                .map_err(|e| CorvidErr::argument(format!("update callback failed: {e}")))?;
            if ret.is_none() {
                coll.delete(&k).map_err(CorvidErr::from)?;
                Ok(())
            } else {
                let doc = value_from_py(&ret)?;
                coll.insert(&k, &doc).map_err(CorvidErr::from)
            }
        })?)
    }

    /// Merge the top-level fields of `patch` into the document at `key`
    /// (creating it if absent).
    fn patch(&self, key: &Bound<'_, PyAny>, patch: &Bound<'_, PyAny>) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            let k = key_from_py(key)?;
            let p = value_from_py(patch)?;
            coll.patch(&k, &p).map_err(CorvidErr::from)
        })?)
    }

    /// Atomically write `replacement` only if the current value equals
    /// `expected` (`None` = must be absent; `replacement=None` deletes
    /// on match). Returns whether the write was applied.
    fn compare_and_set(
        &self,
        key: &Bound<'_, PyAny>,
        expected: &Bound<'_, PyAny>,
        replacement: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        Ok(self.with_coll(|coll| {
            let k = key_from_py(key)?;
            let ex = opt_value(expected)?;
            let re = opt_value(replacement)?;
            coll.compare_and_set(&k, ex.as_ref(), re)
                .map_err(CorvidErr::from)
        })?)
    }

    /// Delete `key`; returns whether it existed.
    fn delete(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.with_coll(|coll| coll.delete(&key_from_py(key)?).map_err(CorvidErr::from))?)
    }

    /// Delete every document matching `pred` (see `field()`); returns
    /// the removed count.
    fn delete_where(&self, pred: &Pred) -> PyResult<u32> {
        Ok(self.with_coll(|coll| {
            coll.delete_where(pred.pred.clone())
                .map(|n| n as u32)
                .map_err(CorvidErr::from)
        })?)
    }

    /// Delete a batch of keys; returns the removed count.
    fn delete_batch(&self, keys: Vec<Bound<'_, PyAny>>) -> PyResult<u32> {
        Ok(self.with_coll(|coll| {
            let mut ks: Vec<Vec<u8>> = Vec::with_capacity(keys.len());
            for k in &keys {
                ks.push(key_from_py(k)?);
            }
            let refs: Vec<&[u8]> = ks.iter().map(|k| k.as_slice()).collect();
            coll.delete_batch(&refs)
                .map(|n| n as u32)
                .map_err(CorvidErr::from)
        })?)
    }

    // -- TTL ----------------------------------------------------------------

    /// Insert with an expiry instant (`expires_at`, epoch units of your
    /// choosing).
    fn insert_with_ttl(
        &self,
        key: &Bound<'_, PyAny>,
        doc: &Bound<'_, PyAny>,
        expires_at: i64,
    ) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            let k = key_from_py(key)?;
            let v = value_from_py(doc)?;
            coll.insert_with_ttl(&k, &v, expires_at)
                .map_err(CorvidErr::from)
        })?)
    }

    /// Set the expiry instant for an existing key.
    fn set_ttl(&self, key: &Bound<'_, PyAny>, expires_at: i64) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            coll.set_ttl(&key_from_py(key)?, expires_at)
                .map_err(CorvidErr::from)
        })?)
    }

    /// The key's expiry instant, or `None` when it has no TTL.
    fn get_ttl(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let ttl = self.with_coll(|coll| coll.ttl(&key_from_py(key)?).map_err(CorvidErr::from))?;
        Ok(match ttl {
            None => py.None(),
            Some(t) => out(py, t)?,
        })
    }

    /// Remove every expired key as of `now`; returns the purged count.
    fn purge_expired(&self, now: i64) -> PyResult<u32> {
        Ok(self.with_coll(|coll| {
            coll.purge_expired(now)
                .map(|n| n as u32)
                .map_err(CorvidErr::from)
        })?)
    }

    // -- reads ----------------------------------------------------------------

    /// The document at `key`, or `None` when absent.
    fn get(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let doc = self.with_coll(|coll| coll.get(&key_from_py(key)?).map_err(CorvidErr::from))?;
        Ok(match doc {
            Some(v) => value_to_py(py, &v)?,
            None => py.None(),
        })
    }

    /// Every `(key, document)` in key order, as a list of tuples.
    fn scan(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>)>> {
        let rows = self.with_coll(|coll| coll.scan().map_err(CorvidErr::from))?;
        let mut out = Vec::with_capacity(rows.len());
        for (k, v) in rows {
            out.push((key_to_py(py, &k)?, value_to_py(py, &v)?));
        }
        Ok(out)
    }

    /// Stream with a callback `fn(key, doc) -> bool` — returning a
    /// falsy value stops the walk early (not an error). Returns the
    /// number of rows visited. The callback must NOT call methods on
    /// this same Collection (non-reentrant handle lock).
    fn scan_each(&self, py: Python<'_>, cb: &Bound<'_, PyAny>) -> PyResult<u32> {
        let mut visited: u32 = 0;
        self.with_coll(|coll| {
            coll.for_each_doc(|key, doc| {
                visited += 1;
                let kj = key_to_py(py, key).map_err(engine_err)?;
                let dj = value_to_py(py, &doc).map_err(engine_err)?;
                let ret = cb.call1((kj, dj)).map_err(|e| {
                    corvid::Error::InvalidArgument(format!("scan callback failed: {e}"))
                })?;
                let cont = ret.is_truthy().map_err(engine_err)?;
                Ok(cont)
            })
            .map_err(CorvidErr::from)
        })?;
        Ok(visited)
    }

    /// Keyset pagination: up to `limit` rows strictly after `after`
    /// (`None` starts at the beginning). Returns a `Page` with `rows`
    /// and the `next` cursor (`None` at the end).
    #[pyo3(signature = (after = None, limit = 10))]
    fn page(
        &self,
        py: Python<'_>,
        after: Option<&Bound<'_, PyAny>>,
        limit: usize,
    ) -> PyResult<Page> {
        let page = self.with_coll(|coll| {
            let after_key = match after {
                None => None,
                Some(ob) if ob.is_none() => None,
                Some(ob) => Some(key_from_py(ob)?),
            };
            coll.page(after_key.as_deref(), limit)
                .map_err(CorvidErr::from)
        })?;
        let mut rows = Vec::with_capacity(page.rows.len());
        for (k, v) in page.rows {
            rows.push((key_to_py(py, &k)?, value_to_py(py, &v)?));
        }
        let next = match page.next {
            Some(k) => Some(key_to_py(py, &k)?),
            None => None,
        };
        Ok(Page { rows, next })
    }

    /// Whether the collection is empty.
    fn is_empty(&self) -> PyResult<bool> {
        Ok(self.with_coll(|coll| coll.is_empty().map_err(CorvidErr::from))?)
    }

    // -- indexes ---------------------------------------------------------------

    /// Create a scalar index over `field`.
    fn create_scalar_index(&self, field: &str) -> PyResult<()> {
        Ok(self.with_coll(|coll| coll.create_scalar_index(field).map_err(CorvidErr::from))?)
    }

    /// Create a compound index over `fields`.
    fn create_compound_index(&self, fields: Vec<String>) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            let refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
            coll.create_compound_index(&refs).map_err(CorvidErr::from)
        })?)
    }

    /// Create an in-memory BM25 inverted-text index over `field`.
    fn create_text_index(&self, field: &str) -> PyResult<()> {
        Ok(self.with_coll(|coll| coll.create_text_index(field).map_err(CorvidErr::from))?)
    }

    /// Create an on-disk BM25 inverted-text index over `field`.
    fn create_text_index_ondisk(&self, field: &str) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            coll.create_text_index_ondisk(field)
                .map_err(CorvidErr::from)
        })?)
    }

    /// Create a geospatial index over the point `field`.
    fn create_geo_index(&self, field: &str) -> PyResult<()> {
        Ok(self.with_coll(|coll| coll.create_geo_index(field).map_err(CorvidErr::from))?)
    }

    /// Create an in-memory HNSW vector index (`metric`:
    /// 'cosine'|'dot'|'l2').
    fn create_vector_index(&self, field: &str, metric: &str) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            coll.create_vector_index(field, parse_metric(metric)?)
                .map_err(CorvidErr::from)
        })?)
    }

    /// Create an in-memory quantized HNSW vector index (`quant`:
    /// 'none'|'binary'|'scalar').
    fn create_vector_index_quantized(
        &self,
        field: &str,
        metric: &str,
        quant: &str,
    ) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            coll.create_vector_index_quantized(field, parse_metric(metric)?, parse_quant(quant)?)
                .map_err(CorvidErr::from)
        })?)
    }

    /// Create an on-disk HNSW vector index.
    fn create_vector_index_ondisk(&self, field: &str, metric: &str) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            coll.create_vector_index_ondisk(field, parse_metric(metric)?)
                .map_err(CorvidErr::from)
        })?)
    }

    /// Create an on-disk quantized HNSW vector index.
    fn create_vector_index_ondisk_quantized(
        &self,
        field: &str,
        metric: &str,
        quant: &str,
    ) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            coll.create_vector_index_ondisk_quantized(
                field,
                parse_metric(metric)?,
                parse_quant(quant)?,
            )
            .map_err(CorvidErr::from)
        })?)
    }

    /// Create an in-memory product-quantized HNSW index (m subspaces,
    /// k centroids; `dim % m == 0` required).
    fn create_vector_index_pq(
        &self,
        field: &str,
        metric: &str,
        m: usize,
        k: usize,
    ) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            coll.create_vector_index_pq(field, parse_metric(metric)?, m, k)
                .map_err(CorvidErr::from)
        })?)
    }

    /// Create an on-disk product-quantized HNSW index.
    fn create_vector_index_ondisk_pq(
        &self,
        field: &str,
        metric: &str,
        m: usize,
        k: usize,
    ) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            coll.create_vector_index_ondisk_pq(field, parse_metric(metric)?, m, k)
                .map_err(CorvidErr::from)
        })?)
    }

    // -- schema ----------------------------------------------------------------

    /// Declare the collection's schema (a list of `SchemaField`s);
    /// replaces any previous one.
    fn set_schema(&self, fields: Vec<PyRef<'_, SchemaField>>) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            let mut schema = Schema::new();
            for f in fields {
                let mut field = Field::new(&f.name, parse_field_type(&f.ty)?);
                if f.required {
                    field = field.required();
                }
                if f.unique {
                    field = field.unique();
                }
                schema = schema.field(field);
            }
            coll.set_schema(&schema).map_err(CorvidErr::from)
        })?)
    }

    /// The declared schema as `SchemaField`s, or `None` when none.
    fn schema(&self) -> PyResult<Option<Vec<SchemaField>>> {
        Ok(self.with_coll(|coll| {
            Ok(coll.schema().map(|s| {
                s.fields()
                    .iter()
                    .map(|f| SchemaField {
                        name: f.name.clone(),
                        ty: field_type_name(f.ty).to_string(),
                        required: f.required,
                        unique: f.unique,
                    })
                    .collect()
            }))
        })?)
    }

    // -- graph ----------------------------------------------------------------

    /// Add a directed edge `from --relation--> to`.
    fn link(&self, from: &Bound<'_, PyAny>, relation: &str, to: &Bound<'_, PyAny>) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            coll.link(&key_from_py(from)?, relation, &key_from_py(to)?)
                .map_err(CorvidErr::from)
        })?)
    }

    /// Add a weighted directed edge.
    fn link_weighted(
        &self,
        from: &Bound<'_, PyAny>,
        relation: &str,
        to: &Bound<'_, PyAny>,
        weight: f64,
    ) -> PyResult<()> {
        Ok(self.with_coll(|coll| {
            coll.link_weighted(&key_from_py(from)?, relation, &key_from_py(to)?, weight)
                .map_err(CorvidErr::from)
        })?)
    }

    /// Remove an edge; returns whether it existed.
    fn unlink(
        &self,
        from: &Bound<'_, PyAny>,
        relation: &str,
        to: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        Ok(self.with_coll(|coll| {
            coll.unlink(&key_from_py(from)?, relation, &key_from_py(to)?)
                .map_err(CorvidErr::from)
        })?)
    }

    /// Out-edge destinations from `from` over `relation`.
    fn neighbors(
        &self,
        py: Python<'_>,
        from: &Bound<'_, PyAny>,
        relation: &str,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let keys = self.with_coll(|coll| {
            coll.neighbors(&key_from_py(from)?, relation)
                .map_err(CorvidErr::from)
        })?;
        keys_to_py(py, keys)
    }

    /// In-edge sources to `to` over `relation`.
    fn in_neighbors(
        &self,
        py: Python<'_>,
        to: &Bound<'_, PyAny>,
        relation: &str,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let keys = self.with_coll(|coll| {
            coll.in_neighbors(&key_from_py(to)?, relation)
                .map_err(CorvidErr::from)
        })?;
        keys_to_py(py, keys)
    }

    /// Weighted out-edges as `(key, weight)` tuples.
    fn neighbors_weighted(
        &self,
        py: Python<'_>,
        from: &Bound<'_, PyAny>,
        relation: &str,
    ) -> PyResult<Vec<(Py<PyAny>, f64)>> {
        let pairs = self.with_coll(|coll| {
            coll.neighbors_weighted(&key_from_py(from)?, relation)
                .map_err(CorvidErr::from)
        })?;
        let mut out = Vec::with_capacity(pairs.len());
        for (k, w) in pairs {
            out.push((key_to_py(py, &k)?, w));
        }
        Ok(out)
    }

    /// BFS `hops` out over `relation` (cycle-safe).
    fn traverse(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        relation: &str,
        hops: usize,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let keys = self.with_coll(|coll| {
            coll.traverse(&key_from_py(start)?, relation, hops)
                .map_err(CorvidErr::from)
        })?;
        keys_to_py(py, keys)
    }

    // -- geo ------------------------------------------------------------------

    /// Radius search, nearest first (ties by key): a list of `GeoHit`s.
    fn geo_within_radius(
        &self,
        py: Python<'_>,
        field: &str,
        lat: f64,
        lon: f64,
        radius_km: f64,
    ) -> PyResult<Vec<GeoHit>> {
        let hits = self.with_coll(|coll| {
            coll.geo_within_radius(field, lat, lon, radius_km)
                .map_err(CorvidErr::from)
        })?;
        geo_hits(py, hits)
    }

    /// Bounding-box search (key order; no center, so distances are the
    /// 0.0 sentinel).
    fn geo_within_bbox(
        &self,
        py: Python<'_>,
        field: &str,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> PyResult<Vec<GeoHit>> {
        // bbox has no center: the engine returns plain rows and the ABI
        // reports the 0.0 distance sentinel — same here.
        let rows = self.with_coll(|coll| {
            coll.geo_within_bbox(field, min_lat, min_lon, max_lat, max_lon)
                .map_err(CorvidErr::from)
        })?;
        let mut out = Vec::with_capacity(rows.len());
        for (k, doc) in rows {
            out.push(GeoHit {
                key: key_to_py(py, &k)?,
                distance_km: 0.0,
                document: value_to_py(py, &doc)?,
            });
        }
        Ok(out)
    }

    /// The `k` nearest points: a list of `GeoHit`s.
    fn geo_nearest(
        &self,
        py: Python<'_>,
        field: &str,
        lat: f64,
        lon: f64,
        k: usize,
    ) -> PyResult<Vec<GeoHit>> {
        let hits = self.with_coll(|coll| {
            coll.geo_nearest(field, lat, lon, k)
                .map_err(CorvidErr::from)
        })?;
        geo_hits(py, hits)
    }

    // -- queries ----------------------------------------------------------------

    /// Begin a fluent query over this collection (one execution per
    /// builder).
    fn query(&self) -> PyResult<crate::QueryPy> {
        let guard = self.inner.lock().expect("collection lock");
        let inner = guard.as_ref().ok_or_else(closed)?;
        Ok(crate::QueryPy::new(
            Arc::clone(&inner.db),
            inner.name.clone(),
            Arc::clone(&inner.counter),
        ))
    }

    /// Release the handle (idempotent); also runs on GC. Derived
    /// handles may outlive the parent Db close.
    fn close(&self) {
        if let Some(inner) = self.inner.lock().expect("collection lock").take() {
            release(&inner.counter);
        }
    }
}

/// Engine geo hits → public `GeoHit` rows (module-level: not a Python
/// method — engine types must not appear in the callable surface).
fn geo_hits(py: Python<'_>, hits: Vec<EngineGeoHit>) -> PyResult<Vec<GeoHit>> {
    let mut out = Vec::with_capacity(hits.len());
    for hit in hits {
        out.push(GeoHit {
            key: key_to_py(py, &hit.key)?,
            distance_km: hit.distance_km,
            document: value_to_py(py, &hit.document)?,
        });
    }
    Ok(out)
}

fn keys_to_py(py: Python<'_>, keys: Vec<Vec<u8>>) -> PyResult<Vec<Py<PyAny>>> {
    let mut out = Vec::with_capacity(keys.len());
    for k in keys {
        out.push(key_to_py(py, &k)?);
    }
    Ok(out)
}

fn opt_value(ob: &Bound<'_, PyAny>) -> CResult<Option<corvid::Value>> {
    if ob.is_none() {
        return Ok(None);
    }
    Ok(Some(value_from_py(ob)?))
}
