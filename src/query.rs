//! The `Query` pyclass — the fluent builder mirroring the engine's
//! `QueryBuilder`. Like the FFI's `QueryHandle`, it stores the
//! builder's parts (`Arc<Db>`, name, filters, sources, knobs) and
//! materializes the real engine `QueryBuilder` exactly once, at the
//! executing call — which CONSUMES the handle (mirroring the engine's
//! by-value `run(self)`). Fluent chaining returns the same object
//! (`q.filter(...).vector(...).run()`); ranking-parameter validation
//! stays at execution, exactly as the engine and the ABI do it.

use std::sync::{Arc, Mutex};

use corvid::filter::Predicate;
use corvid::Metric;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::db::{release, retain, Counter};
use crate::error::{CResult, CorvidErr, ErrCode};
use crate::pred::Pred;
use crate::types::Row;
use crate::value::{key_to_py, value_from_py, value_to_py};

enum Source {
    Vector {
        field: String,
        query: Vec<f32>,
        k: usize,
        metric: Metric,
    },
    Text {
        field: String,
        query: String,
        k: usize,
    },
}

pub(crate) struct QueryInner {
    db: Arc<corvid::Db>,
    name: String,
    counter: Counter,
    filters: Vec<Predicate>,
    sources: Vec<Source>,
    rrf_k: f32,
    mmr_lambda: Option<f32>,
    limit: Option<usize>,
    offset: usize,
    order_by: Option<(String, bool)>,
    projection: Option<Vec<String>>,
    approx: bool,
}

impl QueryInner {
    /// Materialize the engine builder from the stored parts, applying
    /// them in the engine's own builder order. `fuse_rrf` is applied
    /// unconditionally with `rrf_k` (initialized to the engine's
    /// `DEFAULT_RRF_K`), which is identical to the engine's default
    /// fused state.
    fn build(&self) -> corvid::QueryBuilder<'_> {
        let coll = self.db.collection(&self.name);
        let mut b = coll.query();
        for f in &self.filters {
            b = b.filter(f.clone());
        }
        for s in &self.sources {
            match s {
                Source::Vector {
                    field,
                    query,
                    k,
                    metric,
                } => {
                    b = b.vector(field.clone(), query.clone(), *k, *metric);
                }
                Source::Text { field, query, k } => {
                    b = b.text(field.clone(), query.clone(), *k);
                }
            }
        }
        b = b.fuse_rrf(self.rrf_k);
        if let Some(l) = self.mmr_lambda {
            b = b.rerank_mmr(l);
        }
        if self.approx {
            b = b.approx();
        }
        if let Some((field, desc)) = &self.order_by {
            b = b.order_by(field.clone(), *desc);
        }
        if let Some(fields) = &self.projection {
            b = b.select(fields.iter().cloned());
        }
        if self.offset > 0 {
            b = b.offset(self.offset);
        }
        if let Some(n) = self.limit {
            b = b.limit(n);
        }
        b
    }
}

#[pyclass(name = "Query")]
pub struct QueryPy {
    inner: Mutex<Option<QueryInner>>,
}

impl QueryPy {
    pub(crate) fn new(db: Arc<corvid::Db>, name: String, counter: Counter) -> Self {
        // A query is a derived handle: count it until the executing
        // terminal op (or close/drop) releases it — the §4.13 gate.
        retain(&counter);
        Self {
            inner: Mutex::new(Some(QueryInner {
                db,
                name,
                counter,
                filters: Vec::new(),
                sources: Vec::new(),
                rrf_k: corvid::DEFAULT_RRF_K,
                mmr_lambda: None,
                limit: None,
                offset: 0,
                order_by: None,
                projection: None,
                approx: false,
            })),
        }
    }

    fn with<R>(&self, f: impl FnOnce(&mut QueryInner) -> CResult<R>) -> CResult<R> {
        let mut guard = self.inner.lock().expect("query lock");
        match guard.as_mut() {
            Some(inner) => f(inner),
            None => Err(CorvidErr::new(
                ErrCode::Argument,
                "query was already executed or closed",
            )),
        }
    }

    /// Take the inner state (consume) — terminal ops run through this,
    /// releasing the derived-handle counter exactly once.
    fn consume(&self) -> CResult<QueryInner> {
        let mut guard = self.inner.lock().expect("query lock");
        match guard.take() {
            Some(inner) => {
                release(&inner.counter);
                Ok(inner)
            }
            None => Err(CorvidErr::new(
                ErrCode::Argument,
                "query was already executed or closed",
            )),
        }
    }
}

impl Drop for QueryPy {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.lock().expect("query lock").take() {
            release(&inner.counter);
        }
    }
}

#[pymethods]
impl QueryPy {
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(&self, _py: Python<'_>, _exc: Option<Bound<'_, PyAny>>) {
        self.close();
    }

    // -- fluent setters (each returns the same builder) ----------------------

    /// Restrict to documents matching `pred` (multiple filters AND
    /// together).
    fn filter<'a>(slf: PyRef<'a, Self>, pred: &Pred) -> PyResult<PyRef<'a, Self>> {
        slf.with(|inner| {
            inner.filters.push(pred.pred.clone());
            Ok(())
        })?;
        Ok(slf)
    }

    /// Add a vector source over `field` (`query` an `array('f')`),
    /// contributing up to `k` candidates.
    #[pyo3(signature = (field, query, k, metric = "cosine"))]
    fn vector<'a>(
        slf: PyRef<'a, Self>,
        field: &str,
        query: &Bound<'_, PyAny>,
        k: usize,
        metric: &str,
    ) -> PyResult<PyRef<'a, Self>> {
        slf.with(|inner| {
            let v = value_from_py(query)?;
            let elems = match v {
                corvid::Value::Vector(f) => f,
                _ => {
                    return Err(CorvidErr::argument(
                        "query.vector wants an array('f') (float32 vector)",
                    ));
                }
            };
            let m = crate::collection::parse_metric(metric)?;
            inner.sources.push(Source::Vector {
                field: field.to_string(),
                query: elems,
                k,
                metric: m,
            });
            Ok(())
        })?;
        Ok(slf)
    }

    /// Add a BM25 text source over `field`, contributing up to `k`
    /// candidates.
    fn text<'a>(
        slf: PyRef<'a, Self>,
        field: &str,
        query: &str,
        k: usize,
    ) -> PyResult<PyRef<'a, Self>> {
        slf.with(|inner| {
            inner.sources.push(Source::Text {
                field: field.to_string(),
                query: query.to_string(),
                k,
            });
            Ok(())
        })?;
        Ok(slf)
    }

    /// Set the Reciprocal Rank Fusion constant (default 60; validated
    /// at execution).
    fn fuse_rrf<'a>(slf: PyRef<'a, Self>, k: f64) -> PyResult<PyRef<'a, Self>> {
        slf.with(|inner| {
            inner.rrf_k = k as f32;
            Ok(())
        })?;
        Ok(slf)
    }

    /// Rerank fused candidates for diversity (`lambda` in `[0, 1]`;
    /// validated at execution).
    fn rerank_mmr<'a>(slf: PyRef<'a, Self>, lambda: f64) -> PyResult<PyRef<'a, Self>> {
        slf.with(|inner| {
            inner.mmr_lambda = Some(lambda as f32);
            Ok(())
        })?;
        Ok(slf)
    }

    /// Prefer index-backed approximate execution where available.
    fn approx<'a>(slf: PyRef<'a, Self>) -> PyResult<PyRef<'a, Self>> {
        slf.with(|inner| {
            inner.approx = true;
            Ok(())
        })?;
        Ok(slf)
    }

    /// Cap the result count.
    fn limit<'a>(slf: PyRef<'a, Self>, n: usize) -> PyResult<PyRef<'a, Self>> {
        slf.with(|inner| {
            inner.limit = Some(n);
            Ok(())
        })?;
        Ok(slf)
    }

    /// Skip the first `n` results.
    fn offset<'a>(slf: PyRef<'a, Self>, n: usize) -> PyResult<PyRef<'a, Self>> {
        slf.with(|inner| {
            inner.offset = n;
            Ok(())
        })?;
        Ok(slf)
    }

    /// Order by `field` (numbers first in value order, missing-field
    /// rows last, ties by key); `descending` reverses within class
    /// only.
    #[pyo3(signature = (field, descending = false))]
    fn order_by<'a>(
        slf: PyRef<'a, Self>,
        field: &str,
        descending: bool,
    ) -> PyResult<PyRef<'a, Self>> {
        slf.with(|inner| {
            inner.order_by = Some((field.to_string(), descending));
            Ok(())
        })?;
        Ok(slf)
    }

    /// Project results to the named top-level fields.
    fn select<'a>(slf: PyRef<'a, Self>, fields: Vec<String>) -> PyResult<PyRef<'a, Self>> {
        slf.with(|inner| {
            inner.projection = Some(fields);
            Ok(())
        })?;
        Ok(slf)
    }

    // -- terminal (consuming) ops ----------------------------------------------

    /// Execute; rows as `Row` objects (`row.key`, `row.score`,
    /// `row.document`; score 0 for pure filter/order queries).
    /// Consumes the builder.
    fn run(&self, py: Python<'_>) -> PyResult<Vec<Row>> {
        let inner = self.consume()?;
        let rows = inner.build().run().map_err(CorvidErr::from)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push(Row {
                key: key_to_py(py, &row.key)?,
                score: row.score,
                document: value_to_py(py, &row.document)?,
            });
        }
        Ok(out)
    }

    /// Count matching documents (sources/ranking/limit ignored).
    /// Consumes the builder.
    fn count(&self) -> PyResult<usize> {
        let inner = self.consume()?;
        Ok(inner.build().count().map_err(CorvidErr::from)?)
    }

    /// Count distinct values of `field` across the filtered set.
    fn count_distinct(&self, field: &str) -> PyResult<usize> {
        let inner = self.consume()?;
        Ok(inner
            .build()
            .count_distinct(field)
            .map_err(CorvidErr::from)?)
    }

    /// Sum the numeric values of `field` across the filtered set.
    fn sum(&self, field: &str) -> PyResult<f64> {
        let inner = self.consume()?;
        Ok(inner.build().sum(field).map_err(CorvidErr::from)?)
    }

    /// The filtered mean, or `None` when no document has the field.
    fn avg(&self, field: &str) -> PyResult<Option<f64>> {
        let inner = self.consume()?;
        Ok(inner.build().avg(field).map_err(CorvidErr::from)?)
    }

    /// The minimum value of `field`, or `None` when absent everywhere.
    fn min(&self, py: Python<'_>, field: &str) -> PyResult<Py<PyAny>> {
        let inner = self.consume()?;
        let v = inner.build().min(field).map_err(CorvidErr::from)?;
        Ok(match v {
            Some(v) => value_to_py(py, &v)?,
            None => py.None(),
        })
    }

    /// The maximum value of `field`, or `None` when absent everywhere.
    fn max(&self, py: Python<'_>, field: &str) -> PyResult<Py<PyAny>> {
        let inner = self.consume()?;
        let v = inner.build().max(field).map_err(CorvidErr::from)?;
        Ok(match v {
            Some(v) => value_to_py(py, &v)?,
            None => py.None(),
        })
    }

    /// Group counts: `{group_key: count}` in the engine's ascending
    /// order (dicts preserve insertion order). Group keys are the
    /// engine's formatting (text bare, int/float type-tagged `i:1` /
    /// `f:0.5`).
    fn group_count(&self, py: Python<'_>, field: &str) -> PyResult<Py<PyAny>> {
        let inner = self.consume()?;
        let m = inner.build().group_count(field).map_err(CorvidErr::from)?;
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in m {
            dict.set_item(k, v)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Group sums: `{group_key: sum}` over `value_field`, ascending.
    fn group_sum(
        &self,
        py: Python<'_>,
        group_field: &str,
        value_field: &str,
    ) -> PyResult<Py<PyAny>> {
        let inner = self.consume()?;
        let m = inner
            .build()
            .group_sum(group_field, value_field)
            .map_err(CorvidErr::from)?;
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in m {
            dict.set_item(k, v)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Group means: `{group_key: mean}` over `value_field`, ascending.
    fn group_avg(
        &self,
        py: Python<'_>,
        group_field: &str,
        value_field: &str,
    ) -> PyResult<Py<PyAny>> {
        let inner = self.consume()?;
        let m = inner
            .build()
            .group_avg(group_field, value_field)
            .map_err(CorvidErr::from)?;
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in m {
            dict.set_item(k, v)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Abandon the builder without executing (the free path).
    fn close(&self) {
        if let Some(inner) = self.inner.lock().expect("query lock").take() {
            release(&inner.counter);
        }
    }
}
