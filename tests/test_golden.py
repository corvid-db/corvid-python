"""test_golden.py — the golden-suite port for corvid-python.

Replays the engine's committed fixture suite (tests/golden/*.txt —
vendored verbatim from the corvid v0.2.1 release, the same files the
C smoke suite drives) against this binding's public OOP surface
(``corvid``): one OP<TAB>args<TAB>expected line at a time, every line
dispatched, every expectation checked. The fixtures are test-time
inputs — the binding itself parses nothing.

Port conventions (mirroring c/smoke.c in the engine repo and
corvid-node's test/golden.spec.ts):

* ``#`` lines and blank lines are ignored (not counted executable); an
  independent pre-scan counts executable lines so a dispatch loop that
  silently skips a line diverges from ``executed``.
* Value literals: null true false | -123 | 3.5 | inf -inf |
  bits:0x… (f64 from bits) | bits32:0x… (f32) | t(text) | b(bytes)
  | vec(1.5,bits32:0x…,2) | [a,b] | {k=v,k2=v2}.
* Computed doubles (distances, scores, sums) expect ``~x`` (1e-6
  relative tolerance); stored literals compare **bit-exactly** —
  CPython floats are unboxed C doubles and pyo3 copies them by value,
  so NaN payloads, ``-0.0`` and ±inf all survive the boundary with
  their f64 bits intact (the fidelity corner where V8 canonicalizes
  NaN payloads at the N-API boundary; Python has no such caveat, so
  this port compares NaN bit-for-bit rather than as a NaN class).
* Value ops round-trip through a scratch in-memory db (insert + get):
  the Python↔engine value mapping lives inside the native layer, so
  crossing the boundary is what the values.txt lines prove.
"""

import os
import re
import shutil
import struct
import sys
from array import array

import pytest

import corvid
from corvid import Db, SchemaField, and_, field, not_, or_

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
WORK_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
FILES = [
    "values.txt",
    "mutations.txt",
    "queries.txt",
    "schema.txt",
    "graph.txt",
    "geo.txt",
    "persist.txt",
    "admin.txt",
]
TOTAL_LINES = 256


# ---------------------------------------------------------------------------
# Tokenizing
# ---------------------------------------------------------------------------

def split_top(s: str) -> list:
    """Split ``s`` on top-level commas (depth-aware over []{}())."""
    out = []
    depth = 0
    start = 0
    for i in range(len(s) + 1):
        c = s[i] if i < len(s) else ","
        if c in "[{(":
            depth += 1
        elif c in "]})":
            depth -= 1
        if c == "," and depth == 0:
            end = i
            while end > start and s[end - 1] in " \r":
                end -= 1
            if end > start:
                out.append(s[start:end])
            start = i + 1
    return out


def f64_from_bits(bits: int) -> float:
    return struct.unpack("<d", bits.to_bytes(8, "little"))[0]


def f64_bits(n: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", n))[0]


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", bits.to_bytes(4, "little"))[0]


def parse_double(tok: str) -> float:
    """A double token: inf | -inf | nan | bits:0x… | plain."""
    if tok == "inf":
        return float("inf")
    if tok == "-inf":
        return float("-inf")
    if tok == "nan":
        return float("nan")
    if tok.startswith("bits:"):
        return f64_from_bits(int(tok[5:], 16))
    return float(tok)


def double_matches(got: float, tok: str) -> bool:
    """One expected-double token: ``~x`` near; ``=x``/``x``/bits:/inf exact."""
    if tok.startswith("~"):
        return double_near(got, parse_double(tok[1:]))
    return numbers_equal(got, parse_double(tok.lstrip("=")))


def double_near(got: float, want: float) -> bool:
    return abs(got - want) <= 1e-6 * (1 + abs(want))


def numbers_equal(got: float, want: float) -> bool:
    """Bit-exact f64 comparison — NaN payloads included (full fidelity;
    see the module docstring)."""
    return f64_bits(got) == f64_bits(want)


def err_code(expected: str) -> int:
    if not expected.startswith("err:"):
        raise AssertionError(f"error expectation must be err:N, got {expected!r}")
    return int(expected[4:])


def text_body(tok: str) -> str:
    if not (tok.startswith("t(") and tok.endswith(")")):
        raise AssertionError(f"expected a t(...) literal, got {tok!r}")
    return tok[2:-1]


def list_body(tok: str) -> str:
    if not (tok.startswith("k(") and tok.endswith(")")):
        raise AssertionError(f"expected a k(...) list, got {tok!r}")
    return tok[2:-1]


# ---------------------------------------------------------------------------
# Value literals: parse into Python values (the mapping's input form)
# ---------------------------------------------------------------------------

class Cursor:
    def __init__(self, s: str, i: int = 0):
        self.s = s
        self.i = i


def _skip_ws(cur: Cursor) -> None:
    while cur.i < len(cur.s) and cur.s[cur.i] in " \r":
        cur.i += 1


def _delims_after(s: str, at: int, word_len: int) -> bool:
    after = s[at + word_len] if at + word_len < len(s) else None
    return after in (None, ",", "]", "}", " ", "\r")


def _match_bracket(s: str, at: int, open_c: str, close_c: str) -> int:
    depth = 0
    for q in range(at, len(s)):
        if s[q] == open_c:
            depth += 1
        elif s[q] == close_c:
            depth -= 1
            if depth == 0:
                return q
    raise AssertionError(f"unbalanced {open_c}{close_c} in literal")


def _paren_body(s: str, open_at: int) -> str:
    """The body of the (...) starting at ``open_at`` (already past the head)."""
    depth = 0
    for q in range(open_at, len(s)):
        if s[q] == "(":
            depth += 1
        elif s[q] == ")":
            depth -= 1
            if depth == 0:
                return s[open_at + 1 : q]
    raise AssertionError("unbalanced () in literal")


def parse_literal(src: str, cur: Cursor = None):
    """Parse one literal into the Python value the mapping accepts:

    ints → int, floats → float (f64 bits preserved), t() → str,
    b() → bytes, vec() → array('f'), [..] → list, {k=v} → dict.
    """
    if cur is None:
        cur = Cursor(src)
    _skip_ws(cur)
    if cur.i >= len(src):
        raise AssertionError("empty literal")
    s, start = src, cur.i
    c = s[start]

    # numbers: -123 | 3.5 | inf | -inf | nan | bits:0x…
    is_word_num = (
        s.startswith("inf", start)
        or s.startswith("-inf", start)
        or s.startswith("nan", start)
    )
    if c == "-" or c.isdigit() or s.startswith("bits:", start) or is_word_num:
        if is_word_num:
            cur.i = start + (4 if s.startswith("-inf", start) else 3)
            return parse_double(s[start:cur.i])
        j = start
        is_float = False
        is_bits = False
        if s.startswith("bits:", j):
            is_float = True
            is_bits = True
            j += 5
        while j < len(s):
            d = s[j]
            if d.isdigit() or d in "-+":
                j += 1
            elif d in ".eE":
                is_float = True
                j += 1
            elif is_bits and d in "0123456789abcdefABCDEFxX":
                j += 1
            else:
                break
        tok = s[start:j]
        cur.i = j
        if is_bits:
            return f64_from_bits(int(tok[5:], 16))
        if is_float:
            return float(tok)
        return int(tok)  # arbitrary precision; the mapping range-checks i64

    if s.startswith("null", start) and _delims_after(s, start, 4):
        cur.i = start + 4
        return None
    if s.startswith("true", start) and _delims_after(s, start, 4):
        cur.i = start + 4
        return True
    if s.startswith("false", start) and _delims_after(s, start, 5):
        cur.i = start + 5
        return False

    if (c == "t" or c == "b") and s[start + 1 : start + 2] == "(":
        body = _paren_body(s, start + 1)
        cur.i = start + 2 + len(body) + 1
        return body if c == "t" else body.encode("latin-1")
    if c == "v" and s.startswith("vec(", start):
        body = _paren_body(s, start + 3)
        cur.i = start + 4 + len(body) + 1
        elems = [
            f32_from_bits(int(tok[7:], 16)) if tok.startswith("bits32:") else parse_double(tok)
            for tok in split_top(body)
        ]
        return array("f", elems)

    if c == "[":
        close = _match_bracket(s, start, "[", "]")
        body = s[start + 1 : close]
        out = []
        inner = Cursor(body)
        while inner.i < len(body):
            out.append(parse_literal(body, inner))
            _skip_ws(inner)
            if inner.i < len(body) and body[inner.i] == ",":
                inner.i += 1
        cur.i = close + 1
        return out

    if c == "{":
        close = _match_bracket(s, start, "{", "}")
        body = s[start + 1 : close]
        obj = {}
        j = 0
        while j < len(body):
            ke = body.find("=", j)
            if ke < 0:
                raise AssertionError("map literal needs k=v pairs")
            key = body[j:ke].strip()
            inner = Cursor(body, ke + 1)
            obj[key] = parse_literal(body, inner)
            j = inner.i
            while j < len(body) and body[j] in " ,":
                j += 1
        cur.i = close + 1
        return obj

    raise AssertionError(f"unparseable literal at {s[start:start + 24]!r}")


# ---------------------------------------------------------------------------
# Structural comparison (the mapped Python values)
# ---------------------------------------------------------------------------

def values_equal(got, want) -> bool:
    """Bit-exact, type-exact comparison — the mapping is a bijection, so
    a type mismatch (int vs float vs bool) is a real divergence."""
    if type(got) is not type(want):
        return False
    if isinstance(got, float):
        return f64_bits(got) == f64_bits(want)
    if isinstance(got, (int, str, bytes)):
        return got == want
    if isinstance(got, array):
        g, w = got.tolist(), want.tolist()  # exact f32→f64 widening
        return len(g) == len(w) and all(
            f64_bits(a) == f64_bits(b) for a, b in zip(g, w)
        )
    if isinstance(got, list):
        return len(got) == len(want) and all(
            values_equal(a, b) for a, b in zip(got, want)
        )
    if isinstance(got, dict):
        return got.keys() == want.keys() and all(
            values_equal(got[k], want[k]) for k in want
        )
    return got == want


def render(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v} (bits 0x{f64_bits(v):x})"
    if isinstance(v, array):
        return f"vec({','.join(repr(x) for x in v.tolist())})"
    if isinstance(v, bytes):
        return f"b({v!r})"
    if isinstance(v, list):
        return f"[{','.join(render(x) for x in v)}]"
    return repr(v)


def check_value(got, want_tok: str, ctx: str) -> None:
    want = parse_literal(want_tok)
    assert values_equal(got, want), (
        f"{ctx}: value mismatch: got {render(got)}, want {render(want)}"
    )


def walk_path(root, path: str):
    """Walk a child path like a.b.0.c; None when absent (JS's undefined
    propagation: indexing a non-container yields absence, not a crash)."""
    cur = root
    for seg in path.split("."):
        if cur is None:
            return None
        if seg.isdigit() and isinstance(cur, list):
            i = int(seg)
            cur = cur[i] if i < len(cur) else None
        elif isinstance(cur, dict):
            cur = cur.get(seg)
        else:
            return None
    return cur


# ---------------------------------------------------------------------------
# Predicate helpers over the literal grammar
# ---------------------------------------------------------------------------

CMP = {"eq", "ne", "lt", "le", "gt", "ge"}


def cmp_pred(path: str, op: str, val_lit: str):
    assert op in CMP, f"bad comparison op {op!r}"
    return getattr(field(path), op)(parse_literal(val_lit))


# ---------------------------------------------------------------------------
# The scenario
# ---------------------------------------------------------------------------

class Scenario:
    def __init__(self, file_name: str):
        self.file = file_name
        self.db = None
        self.coll = None
        # values.txt runs against no scenario db (the scratch db below is
        # harness-internal: the mapping needs a boundary crossing).
        self.scratch = Db.open_memory()
        self.workdir = ""
        self.db_path = ""
        self.db2_path = ""
        self.dump_path = ""
        self.backup_path = ""
        self.last_auto_id = 0

    def close_coll(self):
        if self.coll is not None:
            self.coll.close()
            self.coll = None

    def close_db(self):
        self.close_coll()
        if self.db is not None:
            self.db.close()
            self.db = None

    def docs(self):
        if self.coll is None:
            if self.db is None:
                raise AssertionError(f"no database open ({self.file})")
            self.coll = self.db.collection("docs")
        return self.coll

    def open_memory(self):
        self.close_db()
        self.db = Db.open_memory()
        self.docs()

    def open_file(self, path: str):
        self.close_db()
        self.db = Db.open(path)
        self.docs()

    def rt(self, lit_tok: str):
        """Round-trip a literal through the engine (the boundary crossing)."""
        coll = self.scratch.collection("v")
        try:
            coll.insert("k", parse_literal(lit_tok))
            return coll.get("k")
        finally:
            coll.close()


def expect_error(fn, code: int, ctx: str):
    try:
        fn()
    except corvid.CorvidError as e:
        assert e.code == code, f"{ctx}: error code {e.code}, want {code} ({e.message})"
        assert isinstance(e.message, str) and len(e.message) > 0, f"{ctx}: error message present"
        return
    except BaseException as e:  # noqa: BLE001
        raise AssertionError(f"{ctx}: threw {e!r} (not a CorvidError)") from e
    raise AssertionError(f"{ctx}: expected a CorvidError with code {code}, nothing threw")


def type_name(v) -> str:
    if v is None:
        return "null"
    t = type(v)
    if t is bool:
        return "bool"
    if t is int:
        return "int"
    if t is float:
        return "float"
    if t is str:
        return "text"
    if t is bytes:
        return "bytes"
    if t is array:
        return "vector"
    if t is list:
        return "array"
    if t is dict:
        return "map"
    raise AssertionError(f"no type name for {render(v)}")


def length_of(v) -> int:
    if isinstance(v, str):
        return len(v)  # code points (ASCII fixtures)
    if isinstance(v, (list, bytes, array)):
        return len(v)
    if isinstance(v, dict):
        return len(v.keys())
    return 0


def parse_metric(s: str) -> str:
    assert s in ("cosine", "dot", "l2"), f"bad metric {s!r}"
    return s


def parse_quant(s: str) -> str:
    assert s in ("none", "binary", "scalar"), f"bad quant {s!r}"
    return s


FIELD_TYPES = ["any", "bool", "int", "float", "text", "bytes", "vector", "array", "map"]


def parse_field_type(s: str) -> str:
    assert s in FIELD_TYPES, f"bad field type {s!r}"
    return s


def row_keys(rows) -> list:
    return [str(r.key) for r in rows]


def check_keys(keys: list, expected: str, ctx: str) -> None:
    want = list_body(expected)
    wanted = split_top(want) if want else []
    assert keys == wanted, f"{ctx}: row keys {keys} != {wanted}"


def check_scores(scores: list, suffix: str, ctx: str) -> None:
    if not suffix:
        return
    if not suffix.startswith("|"):
        raise AssertionError(f"{ctx}: score suffix must start with |")
    body = suffix[1:]
    if not body:
        return
    toks = split_top(body)
    assert len(scores) == len(toks), f"{ctx}: score count {len(scores)} != {len(toks)}"
    for i, tok in enumerate(toks):
        assert double_matches(scores[i], tok), (
            f"{ctx}: row {i} score {scores[i]} vs {tok!r}"
        )


def split_expected(expected: str):
    at = expected.find("|")
    if at < 0:
        return expected, ""
    return expected[:at], expected[at:]


def group_pairs(expected: str):
    if not (expected.startswith("g(") and expected.endswith(")")):
        raise AssertionError(f"group expectation must be g(...), got {expected!r}")
    pairs = []
    for pair in split_top(expected[2:-1]):
        at = pair.find("=")
        if at < 0:
            raise AssertionError(f"group pair needs key=val, got {pair!r}")
        pairs.append((pair[:at], pair[at + 1 :]))
    return pairs


def check_groups(obj: dict, expected: str, ctx: str) -> None:
    pairs = group_pairs(expected)
    got_keys = list(obj.keys())
    assert got_keys == [k for k, _ in pairs], (
        f"{ctx}: group keys {got_keys} != {[k for k, _ in pairs]}"
    )
    for k, v in pairs:
        assert double_matches(obj[k], v), f"{ctx}: group {k!r} value {obj[k]} vs {v!r}"


# ---------------------------------------------------------------------------
# OP dispatch
# ---------------------------------------------------------------------------

def run_line(s: Scenario, op: str, a: list, expected: str, ctx: str) -> None:
    # ---- pure value ops (boundary crossings through the scratch db) ----
    if op == "VERSION":
        assert corvid.ffi_version() == 1, f"{ctx}: FFI version"
        return
    if op == "VTYPE":
        got = type_name(s.rt(a[0]))
        assert got == expected, f"{ctx}: type {got} != {expected}"
        return
    if op == "VLEN":
        got = length_of(s.rt(a[0]))
        assert got == int(expected), f"{ctx}: length {got} != {expected}"
        return
    if op == "VAS_INT":
        # Engine Ints surface as Python ints of arbitrary precision; a
        # Float or Text literal does not (int/float are distinct types).
        got = s.rt(a[0])
        if expected == "fail":
            assert type(got) is not int, f"{ctx}: as_int unexpectedly ok ({render(got)})"
        else:
            assert type(got) is int, f"{ctx}: as_int type {type_name(got)} != int"
            assert f"ok:{got}" == expected, f"{ctx}: as_int"
        return
    if op == "VAS_FLOAT":
        got = s.rt(a[0])
        if expected == "fail":
            assert type(got) is not float, f"{ctx}: as_float unexpectedly ok"
        else:
            assert type(got) is float, f"{ctx}: as_float type {type_name(got)} != float"
            assert double_matches(got, expected[3:]), f"{ctx}: as_float bits"
        return
    if op == "VAS_BOOL":
        got = s.rt(a[0])
        if expected == "fail":
            assert type(got) is not bool, f"{ctx}: as_bool unexpectedly ok"
        else:
            assert type(got) is bool, f"{ctx}: as_bool type {type_name(got)} != bool"
            assert f"ok:{1 if got else 0}" == expected, f"{ctx}: as_bool"
        return
    if op == "VTEXT_REF":
        got = s.rt(a[0])
        assert isinstance(got, str) and got == text_body(expected), f"{ctx}: text bytes differ"
        return
    if op == "VBYTES_REF":
        got = s.rt(a[0])
        want = expected[2:-1].encode("latin-1")
        assert got == want, f"{ctx}: bytes differ ({got!r} != {want!r})"
        return
    if op == "VVECTOR_REF":
        got = s.rt(a[0])
        rebuilt = parse_literal(a[0])
        assert values_equal(got, rebuilt), (
            f"{ctx}: vector bits differ ({render(got)} != {render(rebuilt)})"
        )
        return
    if op in ("VNEST", "VCLONE"):
        # VCLONE round-trips twice: the second materialization is the
        # clone-analog (independent Python objects from the same stored value).
        got = s.rt(a[0])
        if op == "VCLONE":
            s.rt(a[0])
        child = walk_path(got, a[1])
        if expected == "absent":
            assert child is None, f"{ctx}: unexpectedly present ({render(child)})"
        else:
            check_value(child, expected, ctx)
        return
    if op == "VPUSH":
        arr = s.rt(a[0])
        arr.append(parse_literal(a[1]))
        assert len(arr) == int(expected), f"{ctx}: array length"
        return
    if op == "VPUT":
        obj = s.rt(a[0])
        obj[a[1]] = parse_literal(a[2])
        assert len(obj.keys()) == int(expected), f"{ctx}: map size"
        return
    if op == "NULLFREES":
        # Every close() is idempotent — the free(NULL) analog.
        db2 = Db.open_memory()
        c2 = db2.collection("x")
        c2.close()
        c2.close()
        db2.close()
        db2.close()
        return

    # ---- db-required ops ----
    if op == "COLL":
        s.close_coll()
        s.coll = s.db.collection(a[0])
        assert s.coll.name == a[0], f"{ctx}: collection_name round trip"
        return
    if op in ("INSERT", "INSERT_ERR"):
        docs = s.docs()

        def fn():
            docs.insert(a[0], parse_literal(a[1]))

        if op == "INSERT_ERR":
            expect_error(fn, err_code(expected), ctx)
        else:
            fn()
        return
    if op == "LEN":
        assert len(s.docs()) == int(expected), f"{ctx}: len"
        return
    if op in ("GET", "GETFIELD"):
        got = s.docs().get(a[0])
        if op == "GETFIELD":
            assert got is not None, f"{ctx}: GETFIELD on an absent document"
            child = walk_path(got, a[1])
            if expected == "absent":
                assert child is None, f"{ctx}: field unexpectedly present"
            else:
                check_value(child, expected, ctx)
        elif expected == "absent":
            assert got is None, f"{ctx}: expected absence, got {render(got)}"
        else:
            assert got is not None, f"{ctx}: expected a document, got absence"
            check_value(got, expected, ctx)
        return
    if op in ("PUTMANY", "PUTMANY_ROLLBACK"):
        if len(a) % 2 != 0:
            raise AssertionError(f"{ctx}: PUTMANY wants key/literal pairs")
        entries = [(a[i], parse_literal(a[i + 1])) for i in range(0, len(a), 2)]
        docs = s.docs()

        def fn():
            docs.insert_many(entries)

        if op == "PUTMANY_ROLLBACK":
            expect_error(fn, err_code(expected), ctx)
        else:
            fn()
        return
    if op == "INSERT_AUTO":
        key = s.docs().insert_auto(parse_literal(a[0]))
        assert isinstance(key, str) and re.fullmatch(r"\d{20}", key), (
            f"{ctx}: auto key format ({key})"
        )
        id_ = int(key)
        assert s.last_auto_id == 0 or id_ > s.last_auto_id, f"{ctx}: auto id monotonicity"
        s.last_auto_id = id_
        return
    if op == "UPDATE":
        def bump(cur):
            return {"n": (cur["n"] if cur is not None else 0) + 1}

        s.docs().update(a[0], bump)
        return
    if op == "UPDATE_ABORT":
        def abort(_cur):
            raise RuntimeError("abort")

        expect_error(lambda: s.docs().update(a[0], abort), 12, ctx)
        return
    if op == "PATCH":
        s.docs().patch(a[0], parse_literal(a[1]))
        return
    if op == "CAS":
        applied = s.docs().compare_and_set(
            a[0],
            None if a[1] == "absent" else parse_literal(a[1]),
            None if a[2] == "absent" else parse_literal(a[2]),
        )
        assert ("applied:1" if applied else "applied:0") == expected, f"{ctx}: CAS applied"
        return
    if op == "DELETE":
        existed = s.docs().delete(a[0])
        assert ("existed:1" if existed else "existed:0") == expected, f"{ctx}: delete existed"
        return
    if op == "DELETE_WHERE":
        removed = s.docs().delete_where(cmp_pred(a[0], a[1], a[2]))
        assert f"removed:{removed}" == expected, f"{ctx}: removed count"
        return
    if op == "DELETE_IN":
        removed = s.docs().delete_where(
            field(a[0]).in_([parse_literal(t) for t in a[1:]])
        )
        assert f"removed:{removed}" == expected, f"{ctx}: removed count"
        return
    if op == "DELETE_BATCH":
        removed = s.docs().delete_batch(a)
        assert f"removed:{removed}" == expected, f"{ctx}: removed count"
        return
    if op == "INSERT_TTL":
        s.docs().insert_with_ttl(a[0], parse_literal(a[1]), int(a[2]))
        return
    if op == "GET_TTL":
        ttl = s.docs().get_ttl(a[0])
        got = "nottl" if ttl is None else f"ttl:{ttl}"
        assert got == expected, f"{ctx}: ttl {got} != {expected}"
        return
    if op == "SET_TTL":
        s.docs().set_ttl(a[0], int(a[1]))
        return
    if op == "PURGE":
        purged = s.docs().purge_expired(int(a[0]))
        assert f"purged:{purged}" == expected, f"{ctx}: purged count"
        return
    if op in ("SCAN", "SCAN_STOP"):
        stop = int(a[0]) if op == "SCAN_STOP" else 0
        visited = [0]

        def cb(_key, _doc):
            visited[0] += 1
            return not (stop > 0 and visited[0] >= stop)

        n = s.docs().scan_each(cb)
        assert n == int(expected), f"{ctx}: scanned {n} != {expected}"
        return
    if op == "PAGE":
        after = None if a[0] == "-" else a[0]
        page = s.docs().page(after, int(a[1]))
        key_part, suffix = split_expected(expected)
        check_keys([str(k) for k, _ in page.rows], key_part, ctx)
        assert ("|end" if page.next is None else "|more") == suffix, f"{ctx}: page cursor"
        return

    # ---- predicates + queries ----
    def filtered_count(pred) -> int:
        return s.docs().query().filter(pred).count()

    if op == "QF_COUNT":
        assert filtered_count(cmp_pred(a[0], a[1], a[2])) == int(expected), f"{ctx}: filtered count"
        return
    if op == "QF_EXISTS":
        assert filtered_count(field(a[0]).exists()) == int(expected), f"{ctx}: filtered count"
        return
    if op == "QF_BETWEEN":
        pred = field(a[0]).between(parse_literal(a[1]), parse_literal(a[2]))
        assert filtered_count(pred) == int(expected), f"{ctx}: filtered count"
        return
    if op in ("QF_STARTS", "QF_CONTAINS"):
        body = text_body(a[1])
        pred = field(a[0]).starts_with(body) if op == "QF_STARTS" else field(a[0]).contains(body)
        assert filtered_count(pred) == int(expected), f"{ctx}: filtered count"
        return
    if op == "QF_GEO":
        pred = field(a[0]).within_km(parse_double(a[1]), parse_double(a[2]), parse_double(a[3]))
        assert filtered_count(pred) == int(expected), f"{ctx}: filtered count"
        return
    if op in ("QF_AND", "QF_OR"):
        pred = (
            and_(cmp_pred(a[0], a[1], a[2]), cmp_pred(a[3], a[4], a[5]))
            if op == "QF_AND"
            else or_(cmp_pred(a[0], a[1], a[2]), cmp_pred(a[3], a[4], a[5]))
        )
        assert filtered_count(pred) == int(expected), f"{ctx}: filtered count"
        return
    if op == "QF_NOT":
        assert filtered_count(not_(cmp_pred(a[0], a[1], a[2]))) == int(expected), f"{ctx}: filtered count"
        return
    if op == "PRED_FREE":
        # The never-consumed-root free path: the built predicate is plain
        # garbage — building it and dropping it must be a no-op.
        cmp_pred(a[0], a[1], a[2])
        return
    if op == "Q_ABANDON":
        s.docs().query().close()  # the abandoned-builder free path
        return
    if op in ("QVEC", "APPROX"):
        q = s.docs().query()
        if op == "APPROX":
            q.approx()
        q.vector(a[0], parse_literal(a[1]), int(a[2]), "cosine")
        rows = q.run()
        key_part, suffix = split_expected(expected)
        check_keys(row_keys(rows), key_part, ctx)
        check_scores([r.score for r in rows], suffix, ctx)
        return
    if op == "QTEXT":
        rows = s.docs().query().text(a[0], text_body(a[1]), int(a[2])).run()
        check_keys(row_keys(rows), expected, ctx)
        return
    if op in ("HYBRID", "HYBRID_F"):
        tagged = op == "HYBRID_F"
        vk, tk = int(a[2]), int(a[5])
        limit = int(a[7] if tagged else a[6])
        q = s.docs().query()
        q.filter(field("tag").eq(parse_literal(a[6])) if tagged else field("kind").eq("doc"))
        q.vector(a[0], parse_literal(a[1]), vk, "cosine")
        q.text(a[3], text_body(a[4]), tk)
        q.fuse_rrf(60)
        q.rerank_mmr(1.0)
        q.limit(limit)
        rows = q.run()
        key_part, suffix = split_expected(expected)
        check_keys(row_keys(rows), key_part, ctx)
        check_scores([r.score for r in rows], suffix, ctx)
        return
    if op == "ORDER_BY":
        rows = (
            s.docs()
            .query()
            .order_by(a[0], int(a[1]) == 1)
            .offset(int(a[2]))
            .limit(int(a[3]))
            .run()
        )
        check_keys(row_keys(rows), expected, ctx)
        return
    if op == "SELECT":
        if not (a[0].startswith("(") and a[0].endswith(")")):
            raise AssertionError(f"{ctx}: SELECT's first arg must be a (field,...) group")
        fields = split_top(a[0][1:-1])
        want_key = list_body(a[1])
        rows = s.docs().query().select(fields).run()
        row = next((r for r in rows if str(r.key) == want_key), None)
        assert row is not None, f"{ctx}: row {want_key!r} not in the result"
        check_value(row.document, expected, ctx)
        return
    if op == "AGG_COUNT":
        assert s.docs().query().count() == int(expected), f"{ctx}: count"
        return
    if op == "AGG_DISTINCT":
        assert s.docs().query().count_distinct(a[0]) == int(expected), f"{ctx}: countDistinct"
        return
    if op == "AGG_SUM":
        assert double_matches(s.docs().query().sum(a[0]), expected), f"{ctx}: sum"
        return
    if op == "AGG_AVG":
        avg = s.docs().query().avg(a[0])
        if expected == "none":
            assert avg is None, f"{ctx}: avg none"
        else:
            assert double_matches(avg, expected), f"{ctx}: avg"
        return
    if op in ("AGG_MIN", "AGG_MAX"):
        got = s.docs().query().min(a[0]) if op == "AGG_MIN" else s.docs().query().max(a[0])
        if expected == "absent":
            assert got is None, f"{ctx}: expected absence"
        else:
            assert got is not None, f"{ctx}: expected a value"
            check_value(got, expected, ctx)
        return
    if op in ("AGG_GCOUNT", "AGG_GSUM", "AGG_GAVG"):
        q = s.docs().query()
        obj = (
            q.group_count(a[0])
            if op == "AGG_GCOUNT"
            else q.group_sum(a[0], a[1])
            if op == "AGG_GSUM"
            else q.group_avg(a[0], a[1])
        )
        check_groups(obj, expected, ctx)
        return

    # ---- graph ----
    if op == "LINK":
        s.docs().link(a[0], a[1], a[2])
        return
    if op == "LINK_W":
        s.docs().link_weighted(a[0], a[1], a[2], parse_double(a[3]))
        return
    if op == "UNLINK":
        removed = s.docs().unlink(a[0], a[1], a[2])
        assert ("removed:1" if removed else "removed:0") == expected, f"{ctx}: unlink removed"
        return
    if op in ("NEIGHBORS", "IN_NEIGHBORS"):
        keys = (
            s.docs().neighbors(a[0], a[1])
            if op == "NEIGHBORS"
            else s.docs().in_neighbors(a[0], a[1])
        )
        check_keys([str(k) for k in keys], expected, ctx)
        return
    if op == "NEIGHBORS_W":
        pairs = s.docs().neighbors_weighted(a[0], a[1])
        check_groups(dict((str(k), w) for k, w in pairs), expected, ctx)
        return
    if op == "TRAVERSE":
        keys = s.docs().traverse(a[0], a[1], int(a[2]))
        check_keys([str(k) for k in keys], expected, ctx)
        return

    # ---- geo ----
    if op in ("GINSERT", "GINSERT_M"):
        loc = (
            {"lat": parse_double(a[1]), "lon": parse_double(a[2])}
            if op == "GINSERT_M"
            else [parse_double(a[1]), parse_double(a[2])]
        )
        s.docs().insert(a[0], {"loc": loc})
        return
    if op in ("RADIUS", "NEAREST", "BBOX"):
        if op == "RADIUS":
            hits = s.docs().geo_within_radius(a[0], parse_double(a[1]), parse_double(a[2]), parse_double(a[3]))
        elif op == "NEAREST":
            hits = s.docs().geo_nearest(a[0], parse_double(a[1]), parse_double(a[2]), int(a[3]))
        else:
            hits = s.docs().geo_within_bbox(a[0], parse_double(a[1]), parse_double(a[2]), parse_double(a[3]), parse_double(a[4]))
        key_part, suffix = split_expected(expected)
        check_keys([str(h.key) for h in hits], key_part, ctx)
        if suffix:
            toks = split_top(suffix[1:])
            assert len(hits) == len(toks), f"{ctx}: distance count"
            for i, tok in enumerate(toks):
                assert double_matches(hits[i].distance_km, tok), (
                    f"{ctx}: hit {i} distance {hits[i].distance_km} vs {tok!r}"
                )
        return
    if op == "BBOX_ERR":
        expect_error(
            lambda: s.docs().geo_within_bbox(
                a[0], parse_double(a[1]), parse_double(a[2]), parse_double(a[3]), parse_double(a[4])
            ),
            err_code(expected),
            ctx,
        )
        return

    # ---- schema & indexes ----
    if op == "SET_SCHEMA":
        defs = []
        for spec in a:
            name, ty, required, unique = spec.split("#")
            defs.append(
                SchemaField(name, parse_field_type(ty), required == "1", unique == "1")
            )
        s.docs().set_schema(defs)
        return
    if op == "SCHEMA":
        schema = s.docs().schema()
        assert schema is not None, f"{ctx}: a schema must be declared first"
        got = ",".join(
            f"{f.name}/{f.ty}/{1 if f.required else 0}/{1 if f.unique else 0}"
            for f in schema
        )
        assert got == expected, f"{ctx}: schema round trip {got} != {expected}"
        return
    if op == "SCHEMA9":
        names = ["f_any", "f_bool", "f_int", "f_float", "f_text", "f_bytes", "f_vector", "f_array", "f_map"]
        s.docs().set_schema(
            [
                SchemaField(names[i], FIELD_TYPES[i], i == 1, i == 8)
                for i in range(9)
            ]
        )
        schema = s.docs().schema()
        assert schema is not None, f"{ctx}: the 9-field schema must be declared"
        got = ",".join(str(FIELD_TYPES.index(f.ty)) for f in schema)
        assert len(schema) == 9, f"{ctx}: exactly 9 fields"
        assert got == expected, f"{ctx}: schema9 discriminants {got} != {expected}"
        return
    if op == "SCHEMA_ERR":
        expect_error(lambda: s.docs().insert(a[0], parse_literal(a[1])), err_code(expected), ctx)
        return
    if op == "IDX_SCALAR":
        s.docs().create_scalar_index(a[0])
        return
    if op == "IDX_COMPOUND":
        s.docs().create_compound_index(a)
        return
    if op == "IDX_TEXT":
        s.docs().create_text_index(a[0])
        return
    if op == "IDX_TEXT_DISK":
        s.docs().create_text_index_ondisk(a[0])
        return
    if op == "IDX_GEO":
        s.docs().create_geo_index(a[0])
        return
    if op == "IDX_VEC":
        s.docs().create_vector_index(a[0], parse_metric(a[1]))
        return
    if op == "IDX_VEC_Q":
        s.docs().create_vector_index_quantized(a[0], parse_metric(a[1]), parse_quant(a[2]))
        return
    if op == "IDX_VEC_DISK":
        s.docs().create_vector_index_ondisk(a[0], parse_metric(a[1]))
        return
    if op == "IDX_VEC_DISK_Q":
        s.docs().create_vector_index_ondisk_quantized(a[0], parse_metric(a[1]), parse_quant(a[2]))
        return
    if op in ("IDX_PQ", "IDX_PQ_DISK", "IDX_PQ_ERR"):
        def fn():
            if op == "IDX_PQ_DISK":
                s.docs().create_vector_index_ondisk_pq(a[0], parse_metric(a[1]), int(a[2]), int(a[3]))
            else:
                s.docs().create_vector_index_pq(a[0], parse_metric(a[1]), int(a[2]), int(a[3]))

        if op == "IDX_PQ_ERR":
            expect_error(fn, err_code(expected), ctx)
        else:
            fn()
        return

    # ---- admin & persistence ----
    if op == "FILEDB":
        s.open_file(s.db_path)
        return
    if op == "FILEDB2":
        s.open_file(s.db2_path)
        return
    if op == "DUMP":
        s.db.dump_to_path(s.dump_path)
        return
    if op == "LOAD":
        s.db.load_from_path(s.dump_path)
        return
    if op == "LOAD_RENAMES":
        def fn():
            s.db.load_from_path_with_renames(s.dump_path, {a[0]: a[1]})

        if expected.startswith("err:"):
            expect_error(fn, err_code(expected), ctx)
        else:
            fn()
        return
    if op == "COLLECTIONS":
        check_keys([str(k) for k in s.db.collections()], expected, ctx)
        return
    if op == "BACKUP":
        s.db.backup(s.backup_path)
        return
    if op == "BACKUP_DUP":
        expect_error(lambda: s.db.backup(s.backup_path), 17, ctx)
        return
    if op == "COMPACT_BUSY":
        expect_error(lambda: s.db.compact(), 19, ctx)
        return
    if op == "COMPACT":
        s.close_coll()  # quiesce: the derived-handle gate
        s.db.compact()
        s.docs()  # re-acquire for subsequent lines
        return
    if op == "REOPEN":
        path = s.db_path
        s.close_db()
        s.db = Db.open(path)
        s.docs()
        return

    raise AssertionError(f"{ctx}: unknown OP {op!r}")


# ---------------------------------------------------------------------------
# The fixture driver
# ---------------------------------------------------------------------------

def starts_with_db(file_name: str) -> bool:
    return file_name != "values.txt"


def run_scenario(file_name: str) -> None:
    path = os.path.join(GOLDEN_DIR, file_name)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    stem = file_name[: -len(".txt")]
    s = Scenario(file_name)

    # Scratch paths are per-scenario so file-db scenarios sharing one
    # workdir never touch each other's files.
    s.workdir = os.path.join(
        WORK_ROOT, f"{stem}-{os.getpid()}-{struct.pack('<Q', id(s)).hex()[:8]}"
    )
    os.makedirs(s.workdir, exist_ok=True)
    s.db_path = os.path.join(s.workdir, f"{stem}.redb")
    s.db2_path = os.path.join(s.workdir, f"{stem}-2.redb")
    s.dump_path = os.path.join(s.workdir, f"{stem}.dump")
    s.backup_path = os.path.join(s.workdir, f"{stem}.backup.redb")

    try:
        if starts_with_db(file_name):
            s.open_memory()

        # Independent pre-scan of executable lines (blank / '#' skipped).
        counted = [
            line
            for line in text.split("\n")
            if (t := line.rstrip(" \r").lstrip()) and not t.startswith("#")
        ]

        executed = 0
        for raw in text.split("\n"):
            line = raw.rstrip("\r")
            if len(line) == 0 or line[0] == "#":
                continue
            ctx = f"{file_name}:{executed + 1} OP={line.split(chr(9))[0]}"
            # OP \t ARGS \t EXPECTED
            op, args_str, expected = line, "", ""
            tab1 = line.find("\t")
            if tab1 >= 0:
                op = line[:tab1]
                tab2 = line.find("\t", tab1 + 1)
                if tab2 >= 0:
                    args_str = line[tab1 + 1 : tab2]
                    expected = line[tab2 + 1 :]
                else:
                    args_str = line[tab1 + 1 :]
            args = split_top(args_str) if args_str else []
            run_line(s, op, args, expected, ctx)
            executed += 1

        s.close_db()
        s.scratch.close()

        # A dispatch loop that skipped a counted line diverges here
        # instead of silently passing.
        assert executed == len(counted), (
            f"{file_name}: dispatched {executed} lines, pre-scan counted {len(counted)}"
        )
    finally:
        s.close_db()
        s.scratch.close()
        shutil.rmtree(s.workdir, ignore_errors=True)


@pytest.mark.parametrize("file_name", FILES)
def test_golden_suite(file_name):
    run_scenario(file_name)


def test_golden_suite_totals():
    """All 8 fixture files, 256 executable lines — counted by an
    independent pre-scan, so a truncated or extended fixture fails here."""
    total = 0
    for file_name in FILES:
        with open(os.path.join(GOLDEN_DIR, file_name), encoding="utf-8") as f:
            for line in f.read().split("\n"):
                t = line.rstrip(" \r").lstrip()
                if t and not t.startswith("#"):
                    total += 1
    assert total == TOTAL_LINES, f"total executable fixture lines {total} != {TOTAL_LINES}"
