"""test_value_mapping.py — the binding-specific mapping/lifecycle claims.

The golden fixtures pin engine behavior through this binding, but a set
of claims in docs/PLAN.md §4 and §3 are binding-ONLY — no fixture line
can express them. Each test here pins one:

* out-of-i64 Python ints raise a clean code-12 (never silent wraparound
  or an opaque OverflowError);
* the nesting cap matches the ENGINE's decode cap (``MAX_NESTING``):
  one level deeper than the engine can read back is rejected at the
  boundary, at insert time — before an unreadable document is stored;
* cyclic Python input converts to a clean code-12 (the depth walk
  terminates instead of recursing toward stack overflow);
* bytes/bytearray keys go in as raw bytes; keys that are not valid
  UTF-8 come back out as ``bytes`` (UTF-8 ones as ``str``);
* close-then-use on every handle kind (Db, Collection, Query — plus the
  consumed-by-run Query) raises code 12, never segfaults or silently
  no-ops;
* ``field(...).lt(...)`` actually executes (the one comparison op no
  golden fixture line dispatches — QF_AND uses ge/le, everything else
  eq/between/starts/contains);
* one vector query per metric (cosine/dot/l2) — golden QVEC only ever
  runs the cosine default; dot/l2 surface in fixtures solely through
  index creation (schema.txt IDX_VEC*/IDX_VEC_DISK*).
"""

from array import array

import pytest

from corvid import CorvidError, Db, field

ARG = 12  # ErrorCode.INVALID_ARGUMENT


def open_coll(name="t"):
    return Db.open_memory().collection(name)


# -- i64 boundary -------------------------------------------------------------

def test_int_outside_i64_is_clean_code_12():
    c = open_coll()
    for bad in (2**63, -(2**63) - 1, 2**64, -(2**64) - 1):
        with pytest.raises(CorvidError) as ei:
            c.insert("k", bad)
        assert ei.value.code == ARG, bad
    # ...and nothing leaked into the store.
    assert len(c) == 0


def test_int_i64_boundaries_round_trip():
    c = open_coll()
    c.insert("max", 2**63 - 1)
    c.insert("min", -(2**63))
    assert c.get("max") == 2**63 - 1
    assert c.get("min") == -(2**63)
    # nested too (the converter is recursive)
    c.insert("nest", [2**63 - 1, -(2**63)])
    assert c.get("nest") == [2**63 - 1, -(2**63)]


# -- nesting cap (the engine's decode limit, enforced at the boundary) --------

def nested(levels, leaf=0):
    doc = leaf
    for _ in range(levels):
        doc = [doc]
    return doc


def depth_of(doc):
    d = 0
    while isinstance(doc, list):
        d += 1
        doc = doc[0]
    return d


def test_depth_129_rejected_at_insert_not_stored_unreadable():
    # The engine decodes every stored value on read with its own
    # MAX_NESTING=128 cap; a doc one level deeper would store fine and
    # then fail EVERY read with code 8. The binding's converter uses the
    # same cap, so the failure is a clean code-12 at insert — before
    # anything unreadable reaches the store.
    c = open_coll()
    doc = nested(129)
    with pytest.raises(CorvidError) as ei:
        c.insert("deep", doc)
    assert ei.value.code == ARG
    assert len(c) == 0
    assert c.get("deep") is None


def test_depth_128_round_trips():
    c = open_coll()
    doc = nested(128)
    got = (c.insert("deep", doc), c.get("deep"))[1]
    assert got == doc
    assert depth_of(got) == 128


def test_cyclic_containers_are_clean_code_12():
    c = open_coll()
    cyc_list = []
    cyc_list.append(cyc_list)
    with pytest.raises(CorvidError) as ei:
        c.insert("a", cyc_list)
    assert ei.value.code == ARG
    cyc_dict = {}
    cyc_dict["self"] = cyc_dict
    with pytest.raises(CorvidError) as ei:
        c.insert("b", cyc_dict)
    assert ei.value.code == ARG
    assert len(c) == 0


# -- keys in / out --------------------------------------------------------------

def test_bytes_and_bytearray_keys_in():
    c = open_coll()
    c.insert(b"\x01\x02", {"v": 1})
    c.insert(bytearray(b"bt"), {"v": 2})
    assert c.get(b"\x01\x02") == {"v": 1}
    assert c.get(bytearray(b"bt")) == {"v": 2}
    assert len(c) == 2


def test_non_utf8_keys_out_as_bytes():
    c = open_coll()
    c.insert(b"\xff\xfe", {"v": 1})  # not valid UTF-8
    c.insert("plain", {"v": 2})
    rows = dict((k, d) for k, d in c.scan())
    assert rows[b"\xff\xfe"] == {"v": 1}
    assert rows["plain"] == {"v": 2}
    kinds = {k: type(k).__name__ for k in rows}
    assert kinds == {b"\xff\xfe": "bytes", "plain": "str"}


# -- close-then-use on every handle kind ---------------------------------------

def test_db_close_then_use():
    db = Db.open_memory()
    db.close()
    for op in (
        lambda: db.collection("x"),
        lambda: db.collections(),
        lambda: db.compact(),
        lambda: db.backup("/tmp/never"),
    ):
        with pytest.raises(CorvidError) as ei:
            op()
        assert ei.value.code == ARG


def test_collection_close_then_use():
    c = open_coll()
    c.insert("k", {"n": 1})
    c.close()
    for op in (
        lambda: c.insert("k2", {}),
        lambda: c.get("k"),
        lambda: len(c),
        lambda: c.scan(),
        lambda: c.query(),
        lambda: c.delete("k"),
    ):
        with pytest.raises(CorvidError) as ei:
            op()
        assert ei.value.code == ARG


def test_query_close_then_use_and_consumed_by_run():
    c = open_coll()
    c.insert("k", {"n": 1})

    q = c.query()
    q.close()
    for op in (lambda: q.filter(field("n").eq(1)), lambda: q.run()):
        with pytest.raises(CorvidError) as ei:
            op()
        assert ei.value.code == ARG

    q2 = c.query()
    q2.run()  # terminal op consumes the builder
    with pytest.raises(CorvidError) as ei:
        q2.filter(field("n").eq(1))
    assert ei.value.code == ARG


# -- field().lt() executes (no golden fixture line dispatches it) ---------------

def test_lt_predicate_executes():
    c = open_coll()
    for n in range(1, 6):
        c.insert(f"k{n}", {"n": n})
    rows = c.query().filter(field("n").lt(3)).run()
    assert sorted(r.key for r in rows) == ["k1", "k2"]
    # strict less-than: the boundary value is excluded
    rows = c.query().filter(field("n").lt(1)).run()
    assert rows == []
    # and through delete_where (the other predicate consumer)
    assert c.delete_where(field("n").lt(2)) == 1
    assert c.query().filter(field("n").lt(2)).run() == []


# -- one vector query per metric -------------------------------------------------

def test_vector_query_per_metric():
    # A=[1,0], B=[3,4], C=[0,1] vs query [1,0]: the three metrics rank
    # them in three DIFFERENT orders, so each metric is actually
    # exercised (a metric silently ignored would fail at least two).
    # Golden QVEC only runs the cosine default; dot/l2 reach fixtures
    # only via index creation (schema.txt IDX_VEC_Q / IDX_VEC_DISK).
    docs = {"A": [1.0, 0.0], "B": [3.0, 4.0], "C": [0.0, 1.0]}
    expected = {
        "cosine": ["A", "B", "C"],  # sim 1.0, 0.6, 0.0
        "dot": ["B", "A", "C"],  # 3.0, 1.0, 0.0
        "l2": ["A", "C", "B"],  # dist 0.0, sqrt(2), sqrt(20)
    }
    for metric, want in expected.items():
        c = open_coll(f"vec_{metric}")
        for k, v in docs.items():
            c.insert(k, {"v": array("f", v)})
        rows = c.query().vector("v", array("f", [1.0, 0.0]), 3, metric).run()
        got = [r.key for r in rows]
        assert got == want, f"{metric}: {got} != {want}"
