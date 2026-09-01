"""test_context_managers.py — regression: the ``with`` protocol.

The pyo3 ``__exit__`` methods originally took a single ``_exc``
parameter, but Python's with-statement ALWAYS calls ``__exit__`` with
three arguments (exc_type, exc_value, traceback — ``None`` triple on
the clean path), so every ``with Db(...)`` / ``with collection`` /
``with query`` raised ``TypeError`` at block exit. Found by the
examples tour (examples/*.py); this pins the fix on both the clean and
exception paths.
"""

from array import array

import pytest

import corvid


def test_db_context_manager_clean_exit():
    with corvid.Db.open_memory() as db:
        assert db.collection("docs").insert("k", {"a": 1}) is None


def test_db_context_manager_exception_path():
    with pytest.raises(ValueError, match="boom"):
        with corvid.Db.open_memory() as _db:
            raise ValueError("boom")


def test_collection_and_query_context_managers():
    with corvid.Db.open_memory() as db:
        with db.collection("docs") as docs:
            docs.insert("k", {"v": array("f", [1.0, 0.0])})
            with docs.query() as q:  # abandoned without executing
                q.filter(corvid.field("a").eq(1))
            assert len(docs) == 1
