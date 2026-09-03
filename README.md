# corvid-python

Python binding for [corvid](https://github.com/corvid-db/corvid) — an
embedded database with typed values, vector/text/hybrid search, graph
edges, geo, TTL, and schemas. The engine is compiled in (a Rust pyo3
crate pinned to an exact corvid release tag) and exposed as idiomatic
synchronous OOP: `Db`, `Collection`, a fluent `Query` builder, and
`field()` predicates. No SQL, no JSON, no serialization on the data
path — values map natively (see the value mapping below).

Its correctness story is the engine's **golden suite**: the same
267-line fixture files the C ABI smoke harness runs are replayed
against this binding's public API on every CI run
(`tests/test_golden.py`).

## Install

```sh
pip install corvid-python
```

Published to PyPI by the release workflow (maturin-built abi3 wheels,
one `cp311` wheel per platform covering every Python ≥ 3.11; the
matrix ships `linux-x64` / `linux-arm64` / `macos-arm64` /
`windows-x64`). To build from source instead — Python 3.11–3.14 (CI
exercises 3.14/3.13/3.12/3.11), Rust ≥ 1.88, and a C toolchain:

```sh
pip install maturin
maturin develop --release    # into the active venv
```

## Usage

```python
from array import array

from corvid import Db, field

db = Db.open("app.redb")               # or Db.open_memory()
docs = db.collection("docs")

docs.insert("p1", {
    "title": "rust embedded database",
    "kind": "doc",
    "v": array("f", [1.0, 0.0]),
})

# hybrid retrieval: filter + vector + BM25, fused (RRF) + reranked (MMR)
rows = (
    docs.query()
    .filter(field("kind").eq("doc"))
    .vector("v", array("f", [1.0, 0.0]), 10, "cosine")
    .text("title", "rust database", 10)
    .fuse_rrf(60)
    .rerank_mmr(1.0)
    .limit(5)
    .run()
)                                      # [Row(key, score, document), ...]

for row in rows:
    print(row.key, row.score, row.document["title"])

# predicates everywhere (queries and deletes)
docs.delete_where(field("kind").eq("draft"))

# scalar/compound/text/geo/vector indexes (incl. quantized + PQ + on-disk)
docs.create_vector_index("v", "cosine")

# TTL, graph, geo, schema, CAS, bulk writes, dump/backup/compact …
docs.close()
db.close()
```

Every failure raises a native `CorvidError` with the engine error
`code` (the C ABI's frozen 1–19 table, exported as `ErrorCode`) and
the engine `message`. Type stubs ship in-package (`py.typed`) — the
public API is fully typed.

## Examples

Six runnable programs in [`examples/`](examples/) — one per concept,
with deterministic output, executed on every CI leg: the quickstart
(open, insert, kNN), **hybrid** (filter + vector + BM25, RRF fusion,
MMR rerank), **vector-index** (in-memory / on-disk /
binary-quantized HNSW vs exact, across a reopen), **text-search**
(BM25, English + CJK bigram segmentation), **graph**
(link/neighbors/traverse + the delete cascade), and **geo**
(radius / bbox / nearest-k over real coordinates).

```sh
maturin develop && python examples/hybrid.py
```

## Value mapping

| Python | engine |
| --- | --- |
| `None`, `bool`, `str` | Null / Bool / Text |
| `int` | Int (full i64 — out-of-range ints raise code 12) |
| `float` | Float |
| `bytes` / `bytearray` | Bytes |
| `array('f')` | Vector (other typecodes are rejected) |
| `list` / `tuple` | Array |
| `dict` (str keys) | Map |

Reading back: Int → `int` (arbitrary precision — no ±2^53 boundary,
unlike the JS binding's number/BigInt split), Float → `float` with
**f64 bits preserved exactly** — NaN payloads, `-0.0`, and `±inf` all
round-trip bit-exactly (CPython floats are unboxed C doubles; pyo3
copies them by value — the fidelity corner where V8 canonicalizes NaN
payloads at the N-API boundary; Python has no such caveat). Vector →
`array('f')` (f32-exact both directions), Map → `dict` in the
engine's key order. Keys are `str` (UTF-8) or `bytes` (non-UTF-8 keys
come back as `bytes`).

Python marks the Int/Float distinction natively (`2` is an int, `2.0`
a float), so the mapping is a clean bijection — there is no
Int/Float collapse and no typed-float escape hatch (the JS binding
needs `CorvidFloat` for CAS/unique/group-key corners).

## Surface manifest (docs/SURFACE.tsv)

Every construct of the engine's public surface (the radar-enforced list the
engine publishes as `scripts/bindings/surface.tsv` at each release tag) is
resolved in `docs/SURFACE.tsv`: the Python API exposing it plus the test that
proves it (golden fixture line references), or `N/A` + reason where the v1
binding deliberately does not expose it. `scripts/surface-gate.sh` fails CI
when a line is unresolved, a cell is empty, or the N/A count drifts from the
committed baseline — so an engine pin bump that changes the surface lands in
this gate, not in a user's bug report.

## Development

```sh
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop               # build the native extension
pytest tests                  # the golden suite (267 fixture lines)
cargo fmt --check             # + cargo clippy --all-targets -- -D warnings
```

The plan — architecture ruling (engine compiled in via pyo3 vs
Python-side ctypes/cffi FFI), the full OOP surface, the value
contract, and follow-up tasks — is [docs/PLAN.md](docs/PLAN.md).

## License

MIT.
