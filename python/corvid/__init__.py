"""corvid — the Python binding for the corvid embedded database.

The whole public surface is implemented natively (the pyo3 crate in
this repo, built by maturin as ``corvid._native``): the ``Db`` /
``Collection`` / ``Query`` classes, the ``field()`` predicate builder,
and the ``CorvidError`` exception. This package re-exports it; the
type stubs in ``__init__.pyi`` (shipped with ``py.typed``) carry the
full typing.

Engine semantics, the value mapping, and the error-code table are
documented in docs/PLAN.md and README.md in the repository.
"""

from corvid._native import (
    CorvidError,
    Db,
    ErrorCode,
    FieldRef,
    GeoHit,
    Page,
    Predicate,
    Query,
    Row,
    SchemaField,
    and_,
    field,
    ffi_version,
    not_,
    or_,
)

__all__ = [
    "CorvidError",
    "Db",
    "ErrorCode",
    "FieldRef",
    "GeoHit",
    "Page",
    "Predicate",
    "Query",
    "Row",
    "SchemaField",
    "and_",
    "field",
    "ffi_version",
    "not_",
    "or_",
]

__version__ = "0.1.0"
