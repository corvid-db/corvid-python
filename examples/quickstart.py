# quickstart.py — the README tour as a runnable file.
#
# Open an in-memory database, create a collection, insert three small
# documents carrying 2-d embeddings, run a kNN vector query under
# cosine, and print the ranked rows. Context managers close what was
# opened (the handles are also GC/finalizer-backed, but explicit is
# the idiom).
#
# Run: python examples/quickstart.py   (after `maturin develop`)

# docs:begin:quickstart
from array import array

from corvid import Db

with Db.open_memory() as db:
    docs = db.collection("docs")

    docs.insert("p1", {"title": "rust embedded database", "kind": "doc",
                       "v": array("f", [1.0, 0.0])})
    docs.insert("p2", {"title": "python web frameworks", "kind": "doc",
                       "v": array("f", [0.0, 1.0])})
    docs.insert("p3", {"title": "rust again database", "kind": "doc",
                       "v": array("f", [0.9, 0.1])})

    # kNN: the 3 nearest documents to (1, 0) under cosine.
    rows = (
        docs.query()
        .vector("v", array("f", [1.0, 0.0]), 3, "cosine")
        .run()
    )  # [Row(key, score, document), ...]

    for rank, row in enumerate(rows, start=1):
        print(f"{rank}. {row.key} score={row.score:.6f} "
              f"{row.document['title']}")

    docs.close()
# docs:end:quickstart
