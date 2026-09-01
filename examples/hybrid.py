# hybrid.py — the flagship: filter + vector + BM25, RRF fusion, MMR
# rerank, limit.
#
# Hybrid retrieval over a 4-document corpus: a pre-ranking `kind`
# filter, a vector (ANN) source and a BM25 text source, both
# contributing top-2 candidate lists, fused with Reciprocal Rank
# Fusion (k = 60) and reranked for diversity with MMR (lambda = 1.0),
# capped at 2 rows. The printed scores are RRF rank sums: s1 is rank 1
# of both sources (1/61 + 1/61 = 2/61), s3 rank 2 of both (2/62).
#
# Run: python examples/hybrid.py   (after `maturin develop`)

# docs:begin:hybrid
from array import array

from corvid import Db, field

with Db.open_memory() as db:
    docs = db.collection("docs")

    docs.insert("s1", {"kind": "doc", "body": "rust embedded database",
                       "v": array("f", [1.0, 0.0])})
    docs.insert("s2", {"kind": "doc", "body": "python web frameworks",
                       "v": array("f", [0.0, 1.0])})
    docs.insert("s3", {"kind": "doc", "body": "rust again database",
                       "v": array("f", [0.9, 0.1])})
    docs.insert("m1", {"kind": "meta"})  # filtered out below

    # The flagship query: filter + vector + text, RRF + MMR + limit.
    rows = (
        docs.query()
        .filter(field("kind").eq("doc"))
        .vector("v", array("f", [1.0, 0.0]), 2, "cosine")
        .text("body", "rust database", 2)
        .fuse_rrf(60)
        .rerank_mmr(1.0)
        .limit(2)
        .run()
    )  # [Row(key, score, document), ...]

    for rank, row in enumerate(rows, start=1):
        print(f"{rank}. {row.key} score={row.score:.6f} "
              f"{row.document['body']}")

    docs.close()
# docs:end:hybrid
