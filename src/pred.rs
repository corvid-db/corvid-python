//! Predicate builders: `field(path)` returns a [`FieldRefBuilder`]
//! whose comparison methods (`eq`/`gt`/…/`in_`/`between`/`starts_with`/
//! `contains`/`within_km`) produce opaque [`Pred`] objects; `and_`/
//! `or_`/`not_` (Python keywords force the trailing underscores)
//! compose them. The engine `Predicate` is constructed eagerly and
//! stored — no descriptor-parsing layer (unlike the napi binding,
//! which must cross as plain JSON-able objects); engine types never
//! leak because `Pred` is opaque.

use corvid::filter::{field as engine_field, Predicate};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::{CResult, CorvidErr, ErrCode};
use crate::value::value_from_py;

/// An opaque predicate handle (built by `field()…` / `and_` / `or_` /
/// `not_`). Pass to `Query.filter()` or `Collection.delete_where()`.
#[pyclass(name = "Predicate")]
pub struct Pred {
    pub(crate) pred: Predicate,
}

impl Pred {
    pub(crate) fn new(pred: Predicate) -> Self {
        Self { pred }
    }
}

/// A field-path predicate builder: `field('a.b').gt(2)` — the dotted
/// path may descend maps and (by integer segment) arrays.
#[pyclass(name = "FieldRef")]
pub struct FieldRefBuilder {
    path: String,
}

/// Build a predicate over a (dotted) field path. Compose with
/// `and_`/`or_`/`not_`: `and_(field('n').gt(2), field('tag').eq('x'))`.
#[pyfunction]
pub fn field(path: &str) -> PyResult<FieldRefBuilder> {
    Ok(FieldRefBuilder {
        path: path.to_string(),
    })
}

fn value_arg(ob: &Bound<'_, PyAny>) -> CResult<corvid::Value> {
    value_from_py(ob)
}

#[pymethods]
impl FieldRefBuilder {
    /// Field equals `value` (engine semantic equality: `NaN == NaN`,
    /// `-0.0 == 0.0`, numeric interop across Int/Float).
    fn eq(&self, value: &Bound<'_, PyAny>) -> PyResult<Pred> {
        Ok(Pred::new(engine_field(&self.path).eq(value_arg(value)?)))
    }

    /// Field does not equal `value`.
    fn ne(&self, value: &Bound<'_, PyAny>) -> PyResult<Pred> {
        Ok(Pred::new(engine_field(&self.path).ne(value_arg(value)?)))
    }

    /// Field is less than `value`.
    fn lt(&self, value: &Bound<'_, PyAny>) -> PyResult<Pred> {
        Ok(Pred::new(engine_field(&self.path).lt(value_arg(value)?)))
    }

    /// Field is less than or equal to `value`.
    fn le(&self, value: &Bound<'_, PyAny>) -> PyResult<Pred> {
        Ok(Pred::new(engine_field(&self.path).le(value_arg(value)?)))
    }

    /// Field is greater than `value`.
    fn gt(&self, value: &Bound<'_, PyAny>) -> PyResult<Pred> {
        Ok(Pred::new(engine_field(&self.path).gt(value_arg(value)?)))
    }

    /// Field is greater than or equal to `value`.
    fn ge(&self, value: &Bound<'_, PyAny>) -> PyResult<Pred> {
        Ok(Pred::new(engine_field(&self.path).ge(value_arg(value)?)))
    }

    /// The path exists (any value, including None, at the path).
    fn exists(&self) -> Pred {
        Pred::new(engine_field(&self.path).exists())
    }

    /// Field value is one of `values` (`in` is a Python keyword, hence
    /// the trailing underscore).
    fn in_(&self, values: Vec<Bound<'_, PyAny>>) -> PyResult<Pred> {
        let mut parsed = Vec::with_capacity(values.len());
        for v in &values {
            parsed.push(value_arg(v)?);
        }
        Ok(Pred::new(engine_field(&self.path).is_in(parsed)))
    }

    /// Field value is in the inclusive range `[low, high]`.
    fn between(&self, low: &Bound<'_, PyAny>, high: &Bound<'_, PyAny>) -> PyResult<Pred> {
        Ok(Pred::new(
            engine_field(&self.path).between(value_arg(low)?, value_arg(high)?),
        ))
    }

    /// Text field starts with `prefix`.
    fn starts_with(&self, prefix: &str) -> Pred {
        Pred::new(engine_field(&self.path).starts_with(prefix))
    }

    /// Text field contains `substring`.
    fn contains(&self, substring: &str) -> Pred {
        Pred::new(engine_field(&self.path).contains(substring))
    }

    /// Geo point field is within `radius_km` of `(lat, lon)` (haversine).
    fn within_km(&self, lat: f64, lon: f64, radius_km: f64) -> Pred {
        Pred::new(engine_field(&self.path).within_km(lat, lon, radius_km))
    }
}

fn need_one(what: &str) -> CorvidErr {
    CorvidErr::new(
        ErrCode::Argument,
        format!("{what} need at least one predicate"),
    )
}

/// Logical AND of predicates (Python keyword → trailing underscore).
/// Variadic like the JS binding: `and_(field('n').ge(1), field('n').le(3))`.
#[pyfunction]
#[pyo3(signature = (*preds))]
pub fn and_(preds: Vec<PyRef<'_, Pred>>) -> PyResult<Pred> {
    let mut iter = preds.into_iter().map(|p| p.pred.clone());
    let first = iter.next().ok_or_else(|| PyErr::from(need_one("and_()")))?;
    Ok(Pred::new(iter.fold(first, |a, b| a.and(b))))
}

/// Logical OR of predicates (variadic).
#[pyfunction]
#[pyo3(signature = (*preds))]
pub fn or_(preds: Vec<PyRef<'_, Pred>>) -> PyResult<Pred> {
    let mut iter = preds.into_iter().map(|p| p.pred.clone());
    let first = iter.next().ok_or_else(|| PyErr::from(need_one("or_()")))?;
    Ok(Pred::new(iter.fold(first, |a, b| a.or(b))))
}

/// Logical NOT of a predicate.
#[pyfunction]
pub fn not_(pred: &Pred) -> Pred {
    Pred::new(Predicate::Not(Box::new(pred.pred.clone())))
}
