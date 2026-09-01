# vector_index.py — three vector-index families, ANN vs exact.
#
# A file-backed database (the on-disk index is a disk-resident HNSW
# graph persisted inside the db file) with eight 4-d documents. The
# same embedding is stored under three fields so each index family can
# be demonstrated side by side:
#
#   v_mem  — in-memory HNSW              (create_vector_index)
#   v_disk — on-disk HNSW                (create_vector_index_ondisk)
#   v_q    — in-memory binary-quantized   (create_vector_index_quantized)
#
# The exact (streaming-scan) ranking is printed first, then the ANN
# (approx) ranking served by each index. The unquantized indexes
# answer identically to the scan on this corpus; the binary-quantized
# one genuinely diverges — the recall/footprint trade-off quantization
# makes (binary packs each float32 to one sign bit, ~32x smaller).
# Finally the db is closed and reopened: the on-disk graph reloads and
# serves the same ANN answer without a rebuild.
#
# Scores are RRF ranks (1/(60 + rank)) — the lone vector source's row
# score — so they reflect each lane's own ranking.
#
# Run: python examples/vector_index.py   (after `maturin develop`)

import os
import tempfile
from array import array

from corvid import Db

CORPUS = [
    ("k0", [1.0, 0.0, 0.0, 0.0]),  # nearest
    ("k1", [0.95, 0.05, 0.0, 0.0]),
    ("k2", [0.0, 1.0, 0.0, 0.0]),
    ("k3", [0.0, 0.9, 0.1, 0.0]),
    ("k4", [0.0, 0.0, 1.0, 0.0]),
    ("k5", [0.7, 0.7, 0.0, 0.0]),
    ("k6", [0.0, 0.0, 0.0, 1.0]),
    ("k7", [0.98, 0.02, 0.0, 0.0]),
]
PROBE = array("f", [1.0, 0.0, 0.0, 0.0])


def run_query(items, field_name, approx, label):
    q = items.query().vector(field_name, PROBE, 4, "cosine")
    if approx:
        q = q.approx()
    rows = q.run()
    hits = " ".join(f"{r.key}({r.score:.6f})" for r in rows)
    print(f"{label:<38} {hits}")


with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "vectors.redb")

    with Db.open(path) as db:
        items = db.collection("items")
        for key, v in CORPUS:
            vec = array("f", v)
            items.insert(key, {"v_mem": vec, "v_disk": vec, "v_q": vec})
        items.create_vector_index("v_mem", "cosine")
        items.create_vector_index_ondisk("v_disk", "cosine")
        items.create_vector_index_quantized("v_q", "cosine", "binary")

        print("top-4 nearest to (1,0,0,0) under cosine:")
        run_query(items, "v_mem", False, "exact (scan):")
        run_query(items, "v_mem", True, "ann in-memory HNSW:")
        run_query(items, "v_disk", True, "ann on-disk HNSW:")
        run_query(items, "v_q", True, "ann binary-quantized:")
        print("(the quantized lane trades recall for a ~32x smaller index)")
        items.close()

    # Reopen: the on-disk graph reloads (no rebuild) and answers again.
    with Db.open(path) as db:
        items = db.collection("items")
        run_query(items, "v_disk", True, "ann on-disk after reopen:")
        items.close()
