"""Type stubs for the corvid package — the fully typed public API.

The implementations are native (pyo3); this stub is the typing source
of truth, shipped with ``py.typed``.

The value mapping (docs/PLAN.md §4):

==================  ===============  ===================================
Python (in)         engine Value     Python (out)
==================  ===============  ===================================
``None``            Null             ``None``
``bool``            Bool             ``bool``
``int``             Int (full i64)   ``int`` (arbitrary precision)
``float``           Float            ``float`` (f64 bits preserved —
                                     NaN payloads, ``-0.0``, ``±inf``)
``str``             Text             ``str``
``bytes``/``bytes`` Bytes            ``bytes``
``array('f')``      Vector           ``array('f')`` (f32-exact)
``list``/``tuple``  Array            ``list``
``dict``            Map              ``dict`` (engine key order)
==================  ===============  ===================================

Out-of-i64 ints raise ``CorvidError`` code 12 (``ErrorCode.INVALID_ARGUMENT``).
"""

from array import array
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias, Union

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

__version__: str

#: A document key: `str` (UTF-8) or `bytes` (raw; non-UTF-8 keys come back as `bytes`).
Key: TypeAlias = Union[str, bytes]
#: A float32 vector — exactly ``array.array('f', ...)`` (other typecodes are rejected).
Vector: TypeAlias = array[float]
#: Any corvid document value (see the table in the module docstring).
CorvidValue: TypeAlias = Union[
    None,
    bool,
    int,
    float,
    str,
    bytes,
    Vector,
    Sequence["CorvidValue"],
    dict[str, "CorvidValue"],
]
#: All field types a schema may declare.
FieldType: TypeAlias = Literal["any", "bool", "int", "float", "text", "bytes", "vector", "array", "map"]
Metric: TypeAlias = Literal["cosine", "dot", "l2"]
Quantization: TypeAlias = Literal["none", "binary", "scalar"]

#: The C-ABI FFI generation this binding covers (docs/FFI.md §1.3; value 1).
def ffi_version() -> int: ...

class CorvidError(Exception):
    """Every engine failure — carries the C-ABI error ``code`` (1..=19,
    see :class:`ErrorCode`) and the engine ``message``."""

    code: int
    message: str

class ErrorCode:
    """The frozen C-ABI error-code table (docs/FFI.md §1.3) — never renumbered."""

    DATABASE: int
    TRANSACTION: int
    TABLE: int
    STORAGE: int
    COMMIT: int
    SET_DURABILITY: int
    COMPACTION: int
    DECODE: int
    CORRUPT_INDEX: int
    RESERVED_COLLECTION: int
    INVALID_NAME: int
    INVALID_ARGUMENT: int
    INCOMPATIBLE_FORMAT: int
    EMPTY_INDEX_TRAINING: int
    SCHEMA_VIOLATION: int
    INVALID_DUMP: int
    BACKUP_TARGET_EXISTS: int
    IO: int
    BUSY: int

class SchemaField:
    """One declared schema field (``Collection.set_schema`` / ``Collection.schema``)."""

    name: str
    ty: FieldType
    required: bool
    unique: bool

    def __init__(self, name: str, ty: FieldType, required: bool = False, unique: bool = False) -> None: ...

class Row:
    """One query result row (``Query.run``)."""

    key: Key
    #: RRF-fused rank score (``0.0`` for pure filter/order queries).
    score: float
    #: The full (or projected) stored document.
    document: CorvidValue

class Page:
    """One page of keyset-paginated rows (``Collection.page``)."""

    #: The ``(key, doc)`` rows, in key order.
    rows: list[tuple[Key, CorvidValue]]
    #: The resume cursor (pass as the next call's ``after``), or ``None`` at the end.
    next: Key | None

class GeoHit:
    """One geo-search hit (``Collection.geo_within_radius`` / ``geo_within_bbox`` / ``geo_nearest``)."""

    key: Key
    #: Distance from the query center in km (the ``0.0`` sentinel for bbox searches).
    distance_km: float
    document: CorvidValue

class Predicate:
    """An opaque predicate (built by ``field()…`` / ``and_`` / ``or_`` / ``not_``).
    Pass to ``Query.filter()`` or ``Collection.delete_where()``."""

class FieldRef:
    """A field-path predicate builder: ``field('a.b').gt(2)`` — the dotted
    path may descend maps and (by integer segment) arrays."""

    def eq(self, value: CorvidValue) -> Predicate:
        """Field equals ``value`` (engine semantic equality: ``NaN == NaN``,
        ``-0.0 == 0.0``, numeric interop across int/float)."""

    def ne(self, value: CorvidValue) -> Predicate: ...
    def lt(self, value: CorvidValue) -> Predicate: ...
    def le(self, value: CorvidValue) -> Predicate: ...
    def gt(self, value: CorvidValue) -> Predicate: ...
    def ge(self, value: CorvidValue) -> Predicate: ...
    def exists(self) -> Predicate:
        """The path exists (any value, including ``None``, at the path)."""

    def in_(self, values: Sequence[CorvidValue]) -> Predicate: ...
    def between(self, low: CorvidValue, high: CorvidValue) -> Predicate: ...
    def starts_with(self, prefix: str) -> Predicate: ...
    def contains(self, substring: str) -> Predicate: ...
    def within_km(self, lat: float, lon: float, radius_km: float) -> Predicate: ...

def field(path: str) -> FieldRef:
    """Build a predicate over a (dotted) field path."""

def and_(*preds: Predicate) -> Predicate:
    """Logical AND (Python keyword → trailing underscore)."""

def or_(*preds: Predicate) -> Predicate:
    """Logical OR."""

def not_(pred: Predicate) -> Predicate:
    """Logical NOT."""

class Db:
    """A database handle. ``Db('app.redb')`` / ``Db.open(path)`` for a file,
    ``Db()`` / ``Db.open_memory()`` for a private in-memory database."""

    def __init__(self, path: str | None = None) -> None: ...

    @classmethod
    def open(cls, path: str) -> Db: ...
    @classmethod
    def open_memory(cls) -> Db: ...

    def __enter__(self) -> Db: ...
    def __exit__(self, *exc: object) -> None: ...

    def collection(self, name: str) -> Collection:
        """Acquire a collection handle (lazily created by the engine on first
        write; names are validated at write time)."""

    def collections(self) -> list[str]:
        """The names of the database's collections, in engine order."""

    def backup(self, path: str) -> None:
        """Copy the database to ``path`` (which must not already exist)."""

    def dump_to_path(self, path: str) -> None:
        """Dump the whole database (documents, indexes, schemas, TTLs, edges,
        auto-id counters) to ``path``."""

    def load_from_path(self, path: str) -> None:
        """Replay a dump file into this database (merging)."""

    def load_from_path_with_renames(self, path: str, renames: dict[str, str]) -> None:
        """Replay a dump file, renaming collections per ``renames``
        (``{from: to}``; targets validated before the stream is read)."""

    def compact(self) -> bool:
        """Compact the database file. Requires quiescence: every
        ``Collection``/``Query`` derived from this db must be closed (or have
        executed), otherwise a ``Busy`` ``CorvidError`` (code 19) is raised.
        Returns whether any data was moved out."""

    def close(self) -> None:
        """Close the handle (idempotent). Derived handles may legitimately
        outlive it — the engine lives until the last handle drops."""

class Collection:
    """A collection handle (a context manager; ``close()`` is idempotent)."""

    name: str

    def __enter__(self) -> Collection: ...
    def __exit__(self, *exc: object) -> None: ...
    def __len__(self) -> int: ...

    # -- mutations -----------------------------------------------------------
    def insert(self, key: Key, doc: CorvidValue) -> None:
        """Insert (replace) ``doc`` at ``key``."""

    def insert_many(self, entries: Sequence[tuple[Key, CorvidValue]]) -> None:
        """Bulk atomic insert (``put_many``): one transaction; a violating
        pair rolls the whole batch back."""

    def insert_auto(self, doc: CorvidValue) -> Key:
        """Insert with an engine-generated key (20-digit, strictly monotonic
        per collection); returns the key."""

    def update(
        self,
        key: Key,
        fn: Callable[[CorvidValue | None], CorvidValue | None],
    ) -> None:
        """Read-modify-write: ``fn`` receives the current document (or ``None``
        when absent) and returns the new document — ``None`` to delete. A
        raising callback aborts with code 12 and writes nothing. ``fn`` must
        NOT call methods on this same Collection (non-reentrant handle lock)."""

    def patch(self, key: Key, patch: CorvidValue) -> None:
        """Merge the top-level fields of ``patch`` into the document at ``key``
        (creating it if absent)."""

    def compare_and_set(
        self, key: Key, expected: CorvidValue | None, replacement: CorvidValue | None
    ) -> bool:
        """Atomically write ``replacement`` only if the current value equals
        ``expected`` (``None`` = must be absent; ``replacement=None`` deletes
        on match). Equality is the engine's semantic equality. Returns whether
        the write was applied."""

    def delete(self, key: Key) -> bool: ...
    def delete_where(self, pred: Predicate) -> int: ...
    def delete_batch(self, keys: Sequence[Key]) -> int: ...

    # -- TTL -----------------------------------------------------------------
    def insert_with_ttl(self, key: Key, doc: CorvidValue, expires_at: int) -> None:
        """Insert with an expiry instant (``expires_at``, epoch units of your choosing)."""

    def set_ttl(self, key: Key, expires_at: int) -> None: ...
    def get_ttl(self, key: Key) -> int | None: ...
    def purge_expired(self, now: int) -> int: ...

    # -- reads ---------------------------------------------------------------
    def get(self, key: Key) -> CorvidValue | None: ...
    def scan(self) -> list[tuple[Key, CorvidValue]]: ...
    def scan_each(self, cb: Callable[[Key, CorvidValue], bool]) -> int:
        """Stream with a callback ``fn(key, doc) -> bool`` — a falsy return
        stops the walk early (not an error). Returns the rows visited. The
        callback must NOT call methods on this same Collection."""

    def page(self, after: Key | None = None, limit: int = 10) -> Page: ...
    def is_empty(self) -> bool: ...

    # -- indexes ---------------------------------------------------------------
    def create_scalar_index(self, field: str) -> None: ...
    def create_compound_index(self, fields: Sequence[str]) -> None: ...
    def create_text_index(self, field: str) -> None: ...
    def create_text_index_ondisk(self, field: str) -> None: ...
    def create_geo_index(self, field: str) -> None: ...
    def create_vector_index(self, field: str, metric: Metric) -> None: ...
    def create_vector_index_quantized(self, field: str, metric: Metric, quant: Quantization) -> None: ...
    def create_vector_index_ondisk(self, field: str, metric: Metric) -> None: ...
    def create_vector_index_ondisk_quantized(self, field: str, metric: Metric, quant: Quantization) -> None: ...
    def create_vector_index_pq(self, field: str, metric: Metric, m: int, k: int) -> None:
        """In-memory product-quantized HNSW index (``dim % m == 0`` required)."""

    def create_vector_index_ondisk_pq(self, field: str, metric: Metric, m: int, k: int) -> None: ...

    # -- schema ----------------------------------------------------------------
    def set_schema(self, fields: Sequence[SchemaField]) -> None:
        """Declare the collection's schema; replaces any previous one."""

    def schema(self) -> list[SchemaField] | None: ...

    # -- graph -----------------------------------------------------------------
    def link(self, from: Key, relation: str, to: Key) -> None: ...
    def link_weighted(self, from: Key, relation: str, to: Key, weight: float) -> None: ...
    def unlink(self, from: Key, relation: str, to: Key) -> bool: ...
    def neighbors(self, from: Key, relation: str) -> list[Key]: ...
    def in_neighbors(self, to: Key, relation: str) -> list[Key]: ...
    def neighbors_weighted(self, from: Key, relation: str) -> list[tuple[Key, float]]: ...
    def traverse(self, start: Key, relation: str, hops: int) -> list[Key]:
        """BFS ``hops`` out over ``relation`` (cycle-safe)."""

    # -- geo ---------------------------------------------------------------------
    def geo_within_radius(self, field: str, lat: float, lon: float, radius_km: float) -> list[GeoHit]: ...
    def geo_within_bbox(
        self, field: str, min_lat: float, min_lon: float, max_lat: float, max_lon: float
    ) -> list[GeoHit]: ...
    def geo_nearest(self, field: str, lat: float, lon: float, k: int) -> list[GeoHit]: ...

    # -- queries -------------------------------------------------------------------
    def query(self) -> Query:
        """Begin a fluent query over this collection (one execution per builder)."""

    def close(self) -> None:
        """Release the handle (idempotent); also runs on GC."""

class Query:
    """The fluent query builder (mirrors the engine's ``QueryBuilder``).
    Fluent setters return the same builder — ``q.filter(...).vector(...).run()``.
    The terminal ops (``run`` and every aggregation) consume the builder;
    ``close()`` abandons it without executing."""

    def __enter__(self) -> Query: ...
    def __exit__(self, *exc: object) -> None: ...

    def filter(self, pred: Predicate) -> Query:
        """Restrict to documents matching ``pred`` (multiple filters AND together)."""

    def vector(self, field: str, query: Vector, k: int, metric: Metric = "cosine") -> Query:
        """Add a vector source (``query`` an ``array('f')``) contributing up to ``k`` candidates."""

    def text(self, field: str, query: str, k: int) -> Query:
        """Add a BM25 text source contributing up to ``k`` candidates."""

    def fuse_rrf(self, k: float) -> Query:
        """Set the Reciprocal Rank Fusion constant (default 60; validated at execution)."""

    def rerank_mmr(self, lambda: float) -> Query:
        """Rerank fused candidates for diversity (``lambda`` in ``[0, 1]``)."""

    def approx(self) -> Query:
        """Prefer index-backed approximate execution where available."""

    def limit(self, n: int) -> Query: ...
    def offset(self, n: int) -> Query: ...
    def order_by(self, field: str, descending: bool = False) -> Query: ...
    def select(self, fields: Sequence[str]) -> Query:
        """Project results to the named top-level fields."""

    def run(self) -> list[Row]:
        """Execute; rows as :class:`Row` objects (score ``0.0`` for pure
        filter/order queries). Consumes the builder."""

    def count(self) -> int:
        """Count matching documents (sources/ranking/limit ignored). Consumes the builder."""

    def count_distinct(self, field: str) -> int: ...
    def sum(self, field: str) -> float: ...
    def avg(self, field: str) -> float | None: ...
    def min(self, field: str) -> CorvidValue | None: ...
    def max(self, field: str) -> CorvidValue | None: ...
    def group_count(self, field: str) -> dict[str, int]:
        """Group counts in the engine's ascending order. Group keys are the
        engine's formatting (text bare, int/float type-tagged ``i:1``/``f:0.5``)."""

    def group_sum(self, group_field: str, value_field: str) -> dict[str, float]: ...
    def group_avg(self, group_field: str, value_field: str) -> dict[str, float]: ...

    def close(self) -> None:
        """Abandon the builder without executing."""
