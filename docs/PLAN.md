# corvid-python — the Python binding plan

Date: 2026-09-01 · Status: bootstrap complete (this document) ·
Controller plan: `docs/superpowers/plans/2026-08-31-corvid-ffi.md` in
the engine repo (corvid-db/corvid).

corvid-python is the Python binding for
[corvid](https://github.com/corvid-db/corvid), the second binding of
the bindings program (after corvid-node). It follows the locked
program rules: **golden-suite port before ergonomic sugar**, OOP
idiom gate (handles → native classes, FFI symbols never in the public
API), exact engine-tag pinning, sync-first (the engine is sync; async
variants are additive later, decided by the FFI bench).

## 1. What shipped in this bootstrap

| Piece | Where |
| --- | --- |
| The pyo3 crate (engine-binding layer AND idiom layer — see §3) | `src/*.rs` — `DbPy`/`CollectionPy`/`QueryPy`, value mapping, error mapping, predicates |
| The public package | `python/corvid/__init__.py` re-exports `corvid._native`; `__init__.pyi` + `py.typed` carry the full typing |
| The golden-suite port | `tests/test_golden.py` driving `tests/golden/*.txt` (vendored byte-identical from the v0.4.1 release — the pinned engine tag; the same fixtures the C smoke suite and corvid-node run) |
| Packaging | `pyproject.toml` (maturin, one cp311-abi3 wheel per platform), README (install-pending note), LICENSE (MIT), `.gitignore` |
| CI | `.github/workflows/ci.yml` — surface gate + lint + golden suite × 4 platform legs (linux-x64/arm64, macos-arm64, windows-x64) × {3.14, 3.13, 3.12, 3.11} + `maturin build` smoke |

The golden suite: **267/267 fixture lines** across 8 files
(values 42, mutations 70, queries 40, schema 28, graph 20, geo 19,
persist 13, admin 24), every line dispatched and every expectation
checked through the OOP surface, with the same independent pre-scan
discipline as the C and Node harnesses (a skipped line diverges
`executed` from the counted total instead of silently passing; a
totals test pre-scans all files for exactly 267).

## 2. Architecture ruling: Rust pyo3 crate, engine compiled in

Two architectures were on the table:

1. **(chosen)** A Rust pyo3 crate that links the engine crate directly
   (`corvid-db = { git = "https://github.com/corvid-db/corvid.git", tag = "v0.4.1" }`
   — the engine package is `corvid-db` with lib ident `corvid`; bare
   `corvid` on crates.io is an unrelated crate) and exposes the OOP
   surface through native pyo3 classes, built into a Python extension
   module by
   maturin.
2. Python-side FFI (ctypes/cffi against the release `cdylib`
   artifacts, or unpacking wheels of them).

Ruling: **(1)**, for the same reasons as corvid-node's §2 ruling, plus
Python-specific ones: ctypes/cffi lose all engine type information at
the boundary (every signature is hand-declared `argtypes`/`restype`
that can drift), the C ABI's handle/lifecycle/thread rules would be
re-implemented in a language with no compile step to check them, and
the C ABI's cursor-based iteration (`_next` + `last_error` thread
locals) is a poor match for Python's protocol. With the engine
compiled in, the Rust compiler checks the entire
handle/value/predicate surface against the real engine API at build
time, and one release pipeline (maturin + GitHub Actions) produces
prebuilt abi3 wheels for the platform matrix.

The **cdylib release artifacts remain the C/C++ story** (`corvid.h` +
platform libraries, consumed by corvid-c and any C consumer); Python
gets the engine compiled in. All bindings pin the same engine tag and
prove themselves against the same golden fixtures — one behavioral
truth, N native implementations.

Consequence (documented trade-off): a Rust toolchain is needed to
build from source; prebuilt abi3 wheels are the default install path
for consumers once published.

**Sync-first:** every method is synchronous (the engine is sync).
Async variants (`async def` shims, async iteration over scans) are
additive later per the controller plan — decided by the FFI bench,
never a bootstrap concern.

## 3. The OOP surface (v1) — idiom gate compliance (FFI.md §0.3)

pyo3 supports native exception subclasses, properties, context
managers, and fluent chaining natively, so — unlike corvid-node, which
wraps a minimal napi surface in a JS idiom layer (`index.js`) — the
idiom layer is implemented **directly in Rust** and exposed as the
`corvid` package. `python/corvid/__init__.py` only re-exports; there
is no Python-side wrapper logic to drift from the engine. Engine/FFI
types never appear in the public API (§8 gate): `Predicate` and
`FieldRef` are opaque pyclasses, results are native containers.

| ABI handle | Python class | Notes |
| --- | --- | --- |
| `corvid_db*` | `Db` | `Db.open`/`Db.open_memory` (also `Db(path)`), `close()` idempotent, `__enter__`/`__exit__` |
| `corvid_coll*` | `Collection` | mutations, reads, TTL, indexes (all variants), schema, graph, geo, `query()`, `__len__` |
| `corvid_query*` | `Query` | fluent chaining (`filter().vector().text().fuse_rrf().rerank_mmr().limit().run()`); terminal ops (`run` + every aggregation) consume it; `close()` is the abandoned-builder path |
| `corvid_rows*`/`_strs*`/`_geohits*`/`_groupiter*`/`_schemaiter*` | native containers | cursors materialize as `list[Row]`, `list[str]`, `list[GeoHit]`, `dict[str, float]` (engine order — dicts preserve insertion order), `list[SchemaField]` |
| `corvid_value*` | the value mapping | see §4 |
| `corvid_pred*` | `Predicate` (opaque) | built eagerly by `field('a.b').gt(2)` / `and_`/`or_`/`not_` (Python keywords force the trailing underscores; `and_`/`or_` are variadic like the JS binding) |
| status + `last_error_*` | `CorvidError` | a native exception (subclass of `Exception`); `code` carries the C-ABI error number (frozen 0–19 table, exported as `ErrorCode` class attributes), `message` the engine text |

- **Errors**: `create_exception!` (not `extends=PyException`, which
  needs abi3-py312 and this binding ships cp311-abi3 for 3.11+);
  `code`/`message` are attached to the raised instance — CPython
  exceptions carry a per-instance `__dict__`, so `except CorvidError
  as e: e.code` behaves exactly like an attribute defined on the
  class. (The napi binding, by contrast, must smuggle the code
  through the thrown Error's message as JSON.)
- **Dispose**: `close()` everywhere (idempotent — the analog of the
  ABI's free-NULL no-ops), plus `__enter__`/`__exit__` on `Db`,
  `Collection`, and `Query`; GC drops also release (pyclass `Drop`).
- **Compact gate**: `Db.compact()` mirrors the ABI's §4.13 exclusivity
  rule — a derived-handle counter (1 for the db, +1 per live
  `Collection`/`Query`, released by close/consume/GC) must be at
  exactly 1 AND the engine `Arc` solely owned, else `Busy` (19). The
  golden admin.txt lines pin both the busy and the quiescent path.
- **Reentrancy**: `Collection.update`/`scan_each` callbacks run with
  the handle lock held — calling back into the same `Collection`
  deadlocks (the FFI's portable contract, documented in the docstrings).
- **Naming**: snake_case methods (`insert_many`, `delete_where`,
  `compare_and_set`, `create_vector_index_ondisk_pq`,
  `geo_within_radius`, …) — Python idiom, unlike the JS binding's
  camelCase; same underlying engine ops.

## 4. The value mapping (the binding's value contract)

| Python (in) | engine `Value` | engine (out) | Python (out) |
| --- | --- | --- | --- |
| `None` | `Null` | `Null` | `None` |
| `bool` | `Bool` | `Bool` | `bool` |
| `int` | `Int` (full i64) | `Int` | `int` (arbitrary precision) |
| `float` | `Float` | `Float` | `float` (f64 bits preserved) |
| `str` | `Text` | `Text` | `str` |
| `bytes` / `bytearray` | `Bytes` | `Bytes` | `bytes` |
| `array('f')` | `Vector` | `Vector` | `array('f')` |
| `list` / `tuple` | `Array` | `Array` | `list` |
| `dict` (str keys) | `Map` | `Map` | `dict` (engine key order) |

Documented corners:

- **FULL f64 bit fidelity — the advantage over the JS binding.**
  CPython floats are unboxed C doubles and pyo3 copies them by value,
  so **NaN payloads**, `-0.0`, and `±inf` all survive the boundary
  bit-exactly, in both directions. The engine preserves f64 NaN
  payloads, and Python can actually observe them — unlike the JS
  binding, where V8 canonicalizes NaN payloads at the N-API number
  boundary (`0x7ff8000000000001` crosses napi as `0x7ff8000000000000`)
  and the golden port must compare NaN as a class. This port compares
  NaN **bit-for-bit** (`values.txt` VAS_FLOAT pins payload bits, and
  `mutations.txt` CAS pins a stored payload through a read-back). No
  deviation note needed.
- **Int is arbitrary precision on the Python side**: engine `Int` is
  i64; Python ints exceed that, so out-of-i64 input raises a clean
  `CorvidError` code 12 (never silent wraparound), and engine Ints
  read back as plain `int` with no ±2^53 boundary (the JS binding
  needs the number/BigInt split).
- **No Int/Float collapse and no escape hatch**: Python marks the
  distinction natively (`2` is an int, `2.0` a float), so the mapping
  is a clean bijection — `2` maps to engine `Int(2)`, `2.0` to
  `Float(2.0)`, and each reads back as itself. The JS binding needs
  `CorvidFloat` for the CAS/unique/group-key corners; Python needs
  nothing. `bool` is checked before `int` (bool subclasses int).
- **Vector = `array('f')`** — the stdlib float32 analog of the JS
  binding's `Float32Array` (numpy-free, f32-exact both directions;
  f32→f64 widening is lossless). Other `array` typecodes are rejected
  (`'d'` would silently lose f32 semantics). A `list[float]` is an
  engine `Array`, not a Vector — vectors are typed at construction.
- **Depth cap**: both conversion directions cap nesting at the engine's
  own decode limit (`corvid::value::MAX_NESTING` = 128) — values are
  stored encoded and decoded on every read, so a doc deeper than the
  decoder's cap would store fine and then fail EVERY read (code 8).
  Capping the converters at the same limit (same check, same starting
  depth) makes converter-accepted == decodable: cyclic Python input or
  crafted dumps convert to a clean code-12 error at the boundary,
  before anything unreadable is stored
  (tests/test_value_mapping.py::test_depth_129_rejected_at_insert_not_stored_unreadable).
- **Keys** are `str` (UTF-8) or `bytes`/`bytearray` (raw); keys that
  are not valid UTF-8 come back as `bytes`.

## 5. Packaging & wheels

- maturin builds `corvid._native` into the `corvid` package
  (`python-source = "python"`, `module-name = "corvid._native"`).
- One **cp311-abi3 wheel per platform** covers every Python ≥ 3.11
  (`abi3-py311`); the CI matrix builds linux-x64, linux-arm64,
  macos-arm64, windows-x64. (macos-x64 was retired with GitHub's
  macos-13 runners — no x86_64-darwin runner exists, same as
  corvid-node's matrix.)
- The `extension-module` pyo3 feature is enabled only for wheel
  builds (pyproject `[tool.maturin] features`), so `cargo clippy`/
  `cargo check` still link libpython and see the full API.
- `.cargo/config.toml` sets `git-fetch-with-cli` — the engine git dep
  fetches through the git CLI (some environments rewrite https remotes
  to ssh; libgit2 cannot use the ssh-agent/keychain the CLI uses).
- **Not published to PyPI yet** (the corvid-node M3 lesson): README
  carries the install-pending note, `maturin build` is smoke-tested in
  CI, and publishing waits on the first release tag.

## 6. Follow-up tasks (post-bootstrap)

1. **Publish wiring**: `maturin publish`/`twine upload` from a release
   workflow for the 4 platform wheels; tag `v0.1.0`; verify
   `pip install corvid-python` resolves prebuilt on each platform;
   remove the README install-pending note.
2. **Ergonomic sugar** (only now, per the golden-before-sugar rule):
   a `SchemaBuilder` fluent form; iterator protocol over `scan()`/
   `run()` results if benchmarks justify it; `__iter__` sugar on Page.
3. **API doc pass**: doc comments → sphinx/mkdocs publishing the
   `__init__.pyi` narratives; the full-fidelity value-mapping note
   (§4) prominent.
4. **Bench parity**: port the FFI bench shapes (put/get/scan/hybrid
   through Python vs the engine's native numbers) to quantify the
   pyo3 crossing cost — feeds the async-variants decision.
5. Bump automation: a scripted PR flow that moves the engine git-dep
   tag together with the wheel version (the program's version rule).
6. free-threaded (3.13+ `--disable-gil`) build variant if demand
   appears; musl wheels if demanded.

## 7. Decision log

| Decision | Rationale |
| --- | --- |
| Rust pyo3 crate, engine compiled in (not ctypes/cffi) | §2 above |
| Idiom layer in Rust directly (no Python wrapper module) | pyo3 supports exceptions/properties/context managers/chaining natively; one implementation, no drift; `__init__.py` only re-exports; FFI/engine types never leak |
| Handwritten `__init__.pyi` + `py.typed` | pyo3's stub generation (`pyo3-stub-gen`) is not mature enough to be the typing source of truth; the stub is |
| `array('f')` for Vector (not `list[float]`, not numpy) | f32-exact like the JS binding's `Float32Array`; stdlib-only (no numpy dependency on the data path); lists are engine Arrays — vectors stay typed |
| NaN compared bit-for-bit in the golden port | Python floats round-trip f64 exactly — the fidelity corner V8 fails; pinned by values.txt/mutations.txt |
| `create_exception!` + instance attributes for `CorvidError.code` | attribute-carrying exception subclasses need abi3-py312; cp311-abi3 keeps 3.11 support with identical consumer ergonomics |
| Variadic `and_`/`or_` (trailing underscores) | Python keywords; mirrors the JS binding's variadic `and`/`or` |
| Counter + Arc exclusivity for `compact` | mirrors the ABI §4.13 gate exactly; pinned by admin.txt |
| pyo3 0.29, maturin 1.x | current crates.io stable lines |
| Vendored golden fixtures (byte-identical to the v0.4.1 tag — the pinned engine version) | stable text; the suite must run offline and per-PR (verified identical to `crates/corvid-ffi/golden/`) |
