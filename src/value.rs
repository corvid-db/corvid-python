//! The Python ↔ engine `Value` mapping (docs/PLAN.md §4 — the binding's
//! value contract).
//!
//! Python → engine:
//! - `None`                      → `Null`
//! - `bool`                      → `Bool` (checked before `int`: bool is
//!   an int subclass in Python)
//! - `int`                       → `Int` (full i64; Python ints are
//!   arbitrary precision, so out-of-i64 values raise a clean
//!   InvalidArgument instead of silently wrapping)
//! - `float`                     → `Float` with **f64 bits preserved**
//!   (CPython floats are unboxed C doubles and pyo3 copies them by
//!   value, so NaN payloads, `-0.0`, and `±inf` survive the round trip
//!   bit-exactly — the fidelity corner where V8 canonicalizes NaN at
//!   the N-API boundary; Python has no such caveat)
//! - `str`                       → `Text`
//! - `bytes` / `bytearray`       → `Bytes` (copied)
//! - `array.array('f', …)`       → `Vector` (copied; elements narrowed
//!   to f32 — exact for f32-origin values, NaN payloads included)
//! - `list` / `tuple`            → `Array` (recursive)
//! - `dict` (str keys)           → `Map` (recursive)
//!
//! engine → Python:
//! - `Int`   → `int` (arbitrary precision — no ±2^53 boundary, unlike
//!   the JS binding's number/BigInt split)
//! - `Float` → `float` with f64 bits preserved (NaN payloads included)
//! - `Bytes` → `bytes`, `Vector` → `array('f')` (the stdlib float32
//!   array — numpy-free, f32-exact both directions), `Map` → `dict`
//!   (keys in the engine's sorted order), `Array` → `list`
//!
//! Python marks the Int/Float distinction natively (`2` is an int,
//! `2.0` is a float), so the mapping is a clean bijection with **no
//! collapse and no typed-float escape hatch** — `2` maps to engine
//! `Int(2)`, `2.0` to engine `Float(2.0)`, and each reads back as
//! itself (the JS binding needs `CorvidFloat` for this corner).
//!
//! Both directions carry a nesting-depth cap (`MAX_DEPTH`), set to the
//! engine's own decode limit (`corvid::value::MAX_NESTING`): values are
//! stored encoded and decoded on every read, so anything the converter
//! accepted but the decoder rejects would store fine and then fail
//! EVERY read. Deeper values (or cyclic Python input) convert to a
//! clean InvalidArgument error rather than recursing toward a stack
//! overflow — rejected at the boundary, before storage.

use std::collections::BTreeMap;

use corvid::Value;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString};

use crate::error::{CResult, CorvidErr, ErrCode};

/// Maximum container nesting the converters will walk — the engine's
/// decode cap re-used verbatim (`corvid::value::MAX_NESTING`, engine
/// src/value.rs: `Value::decode` rejects `depth > MAX_NESTING`, so a
/// deeper doc would store fine and fail every read with code 8). Both
/// directions check `depth > MAX_DEPTH` at the same starting depth the
/// decoder does, so converter-accepted == decodable, exactly. Deeper
/// input (including cyclic Python containers, which are
/// depth-unbounded) maps to a clean InvalidArgument at the boundary;
/// engine values deeper than this (only constructible via a crafted
/// dump replay) fail the same way on the way out.
const MAX_DEPTH: usize = corvid::value::MAX_NESTING;

fn argument(msg: &str) -> CorvidErr {
    CorvidErr::new(ErrCode::Argument, msg)
}

fn too_deep() -> CorvidErr {
    argument(&format!(
        "value nesting exceeds the maximum depth of {MAX_DEPTH}"
    ))
}

/// Lift a pyo3/Python error into `CorvidErr` (code 12 — argument): the
/// conversion itself failed (wrong type, non-string dict key, ...).
/// `PyErr`'s `Display` renders the exception, which is the useful part
/// for a conversion failure.
fn py_wrap(e: PyErr) -> CorvidErr {
    CorvidErr {
        code: ErrCode::Argument.num(),
        message: e.to_string(),
    }
}

/// Wrap any `IntoPyObject` into an owned `Py<PyAny>` (pyo3's
/// `IntoPyObjectExt::into_py_any`, wrapped so call sites can use
/// `out(py, v)` uniformly).
pub(crate) fn out<'py, T>(py: Python<'py>, v: T) -> PyResult<Py<PyAny>>
where
    T: pyo3::conversion::IntoPyObject<'py>,
    T::Error: Into<PyErr>,
{
    use pyo3::conversion::IntoPyObjectExt;
    v.into_py_any(py)
}

/// Convert an arbitrary Python value into an engine `Value`.
pub fn value_from_py(ob: &Bound<'_, PyAny>) -> CResult<Value> {
    value_from_py_at(ob, 0)
}

fn value_from_py_at(ob: &Bound<'_, PyAny>, depth: usize) -> CResult<Value> {
    if depth > MAX_DEPTH {
        return Err(too_deep());
    }

    // None (an exact singleton — not a subclass in Python).
    if ob.is_none() {
        return Ok(Value::Null);
    }

    // bool BEFORE int: bool subclasses int in Python (and bool is
    // final, so the exact cast is sound).
    if let Ok(b) = ob.cast::<pyo3::types::PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }

    if ob.is_instance_of::<PyString>() {
        let s: String = ob.extract().map_err(py_wrap)?;
        return Ok(Value::Text(s));
    }

    if ob.is_instance_of::<PyFloat>() {
        // Bit-exact f64: CPython stores a raw double; extract copies it.
        let f: f64 = ob.extract().map_err(py_wrap)?;
        return Ok(Value::Float(f));
    }

    if ob.is_instance_of::<PyInt>() {
        // Arbitrary-precision ints: only ±i64 is representable; a
        // clean InvalidArgument (12) instead of an opaque OverflowError
        // (an i64 extract of a PyInt can only overflow — the bool and
        // float cases already returned above).
        match ob.extract::<i64>() {
            Ok(i) => Ok(Value::Int(i)),
            Err(_) => Err(argument(
                "int is outside the i64 range (engine Ints are 64-bit)",
            )),
        }
    } else if ob.is_instance_of::<PyBytes>() {
        let b: Vec<u8> = ob.extract().map_err(py_wrap)?;
        Ok(Value::Bytes(b))
    } else if ob.is_instance_of::<PyByteArray>() {
        let b = ob.cast::<PyByteArray>().expect("bytearray").to_vec();
        Ok(Value::Bytes(b))
    } else if is_f32_array(ob)? {
        // Vector: exactly array.array with typecode 'f' (the stdlib
        // Float32Array analog — numpy-free). Other typecodes are
        // rejected: 'd' would silently lose f32 semantics.
        let elems = f32_array_elems(ob)?;
        Ok(Value::Vector(elems))
    } else if ob.is_instance_of::<PyList>() || ob.is_instance_of::<pyo3::types::PyTuple>() {
        // Array: list or tuple (exactly — not arbitrary sequences,
        // which would make dict.keys()/sets silently map).
        let items: Vec<Bound<'_, PyAny>> = ob.extract().map_err(py_wrap)?;
        let mut converted = Vec::with_capacity(items.len());
        for item in &items {
            converted.push(value_from_py_at(item, depth + 1)?);
        }
        Ok(Value::Array(converted))
    } else if ob.is_instance_of::<PyDict>() {
        // Map: dict with string keys (engine maps are string-keyed).
        let dict = ob.cast::<PyDict>().expect("dict");
        let mut map = BTreeMap::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract().map_err(|_| {
                argument("dict keys must be strings (engine maps are string-keyed)")
            })?;
            map.insert(key, value_from_py_at(&v, depth + 1)?);
        }
        Ok(Value::Map(map))
    } else {
        let ty = ob.get_type().name().map_err(py_wrap)?.to_string();
        Err(argument(&format!("unsupported Python value kind: {ty}")))
    }
}

/// Convert an engine `Value` into a Python value (see the module docs
/// for the mapping and its fidelity notes).
pub fn value_to_py(py: Python<'_>, v: &Value) -> PyResult<Py<PyAny>> {
    value_to_py_at(py, v, 0)
}

fn value_to_py_at(py: Python<'_>, v: &Value, depth: usize) -> PyResult<Py<PyAny>> {
    if depth > MAX_DEPTH {
        return Err(PyErr::from(too_deep()));
    }
    match v {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => out(py, *b),
        // Arbitrary precision for free — int(i64) is exact.
        Value::Int(i) => out(py, *i),
        // Unboxed f64 copy: NaN payloads, -0.0, ±inf all survive.
        Value::Float(f) => out(py, PyFloat::new(py, *f)),
        Value::Text(s) => out(py, PyString::new(py, s)),
        Value::Bytes(b) => out(py, PyBytes::new(py, b)),
        Value::Vector(f32s) => vector_to_py(py, f32s),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(value_to_py_at(py, item, depth + 1)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Map(map) => {
            let dict = PyDict::new(py);
            for (k, val) in map {
                dict.set_item(k, value_to_py_at(py, val, depth + 1)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

/// engine Vector → `array('f')`: the exact float32 analog of the JS
/// binding's Float32Array. f32→f64 widening is lossless (every f32 is
/// representable in f64, NaN payload high bits included), so values
/// that originated as f32 round-trip bit-exactly.
fn vector_to_py(py: Python<'_>, f32s: &[f32]) -> PyResult<Py<PyAny>> {
    let array_type = pyo3::types::PyModule::import(py, "array")?.getattr("array")?;
    let list = PyList::new(
        py,
        f32s.iter().map(|f| *f as f64), // exact widening
    )?;
    Ok(array_type.call(("f", list), None)?.into_any().unbind())
}

/// Is this an `array.array` with typecode 'f'?
fn is_f32_array(ob: &Bound<'_, PyAny>) -> CResult<bool> {
    let py = ob.py();
    let array_type = pyo3::types::PyModule::import(py, "array")
        .map_err(py_wrap)?
        .getattr("array")
        .map_err(py_wrap)?;
    if !ob.is_instance(&array_type).map_err(py_wrap)? {
        return Ok(false);
    }
    let typecode: String = ob
        .getattr("typecode")
        .map_err(py_wrap)?
        .extract()
        .map_err(py_wrap)?;
    if typecode != "f" {
        return Err(argument(
            "vectors must be array('f') (float32); other array typecodes are not engine Vectors",
        ));
    }
    Ok(true)
}

fn f32_array_elems(ob: &Bound<'_, PyAny>) -> CResult<Vec<f32>> {
    // tolist() gives Python floats (exact f64 widenings of the stored
    // f32s); narrowing back to f32 is exact for f32-origin values.
    let list = ob.call_method0("tolist").map_err(py_wrap)?;
    let elems: Vec<f64> = list.extract().map_err(py_wrap)?;
    Ok(elems.iter().map(|f| *f as f32).collect())
}

/// A key: `str` (UTF-8 encoded) or `bytes`/`bytearray` (raw bytes).
pub fn key_from_py(ob: &Bound<'_, PyAny>) -> CResult<Vec<u8>> {
    if ob.is_instance_of::<PyString>() {
        let s: String = ob.extract().map_err(py_wrap)?;
        return Ok(s.into_bytes());
    }
    if ob.is_instance_of::<PyBytes>() {
        return ob.extract().map_err(py_wrap);
    }
    if ob.is_instance_of::<PyByteArray>() {
        return Ok(ob.cast::<PyByteArray>().expect("bytearray").to_vec());
    }
    Err(argument("keys must be str or bytes"))
}

/// Keys out: valid UTF-8 → str, anything else → bytes.
pub fn key_to_py(py: Python<'_>, k: &[u8]) -> PyResult<Py<PyAny>> {
    match std::str::from_utf8(k) {
        Ok(s) => out(py, PyString::new(py, s)),
        Err(_) => out(py, PyBytes::new(py, k)),
    }
}
