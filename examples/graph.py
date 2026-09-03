# graph.py — directed edges over a small corpus, and delete cascade.
#
# Three documents (ga, gb, gc) linked by a `parent_of` relation, plus
# one edge pointing at `gd` which never exists as a document (dangling
# edges are allowed), and a weighted `route` relation. Demonstrates
# neighbors (key order), in_neighbors, weighted neighbors, BFS traverse
# at 1 and 2 hops (cycle-safe), and the delete cascade: deleting a key
# removes its edges in the same transaction — deleting the never-a-
# document `gd` still drops the `gb -> gd` edge (spec §4.8/§4.11).
#
# Run: python examples/graph.py   (after `maturin develop`)

# docs:begin:graph
from corvid import Db

with Db.open_memory() as db:
    nodes = db.collection("nodes")
    for key in ("ga", "gb", "gc"):
        nodes.insert(key, {"n": key})

    nodes.link("ga", "parent_of", "gb")
    nodes.link("ga", "parent_of", "gc")
    nodes.link("gb", "parent_of", "gd")  # gd never exists as a document
    nodes.link_weighted("ga", "route", "gb", 2.5)
    nodes.link_weighted("ga", "route", "gd", 0.75)

    def show(label, keys):
        print(f"{label:<36} [{' '.join(keys)}]")

    show("neighbors(ga)", nodes.neighbors("ga", "parent_of"))
    show("in_neighbors(gb)", nodes.in_neighbors("gb", "parent_of"))
    routes = " ".join(f"{k}={w:.2f}" for k, w in nodes.neighbors_weighted("ga", "route"))
    print(f"{'routes from ga (weighted):':<36} [{routes}]")
    show("traverse(ga, 1 hop)", nodes.traverse("ga", "parent_of", 1))
    show("traverse(ga, 2 hops)", nodes.traverse("ga", "parent_of", 2))

    # Delete cascade: remove gc (a document) and gd (never a document).
    print("delete gc: existed=", nodes.delete("gc"))
    print("delete gd: existed=", nodes.delete("gd"),
          "(never a document; its edges still cascade)")

    show("neighbors(ga) after deletes", nodes.neighbors("ga", "parent_of"))
    show("neighbors(gb) after deletes", nodes.neighbors("gb", "parent_of"))
    show("traverse(ga, 2 hops) after", nodes.traverse("ga", "parent_of", 2))

    nodes.close()
# docs:end:graph
