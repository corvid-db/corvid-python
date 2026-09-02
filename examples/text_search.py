# text_search.py — BM25 ranking, English and CJK.
#
# Six notes (three English, three CJK) searched through a text index
# with the query builder's BM25 source. Row scores are RRF ranks
# (1/(60 + rank)); the *order* is the BM25 ranking.
#
# The CJK strings exercise the engine's dictionary-free CJK
# segmentation: maximal runs of CJK characters are tokenized as
# sliding BIGRAMS (「东京」… → "东京", …), so an unsegmented CJK query
# matches by its bigrams — "城市" (city) matches both city notes,
# "数据库" (database) matches the ML note.
#
# Phrase matching: engine v0.3.0 added the DIRECT positional search
# to the ABI (consecutive in-order analyzed tokens, stop words
# collapsing out of adjacency), surfaced here as phrase_search() —
# Row.score is the BM25 phrase sum, not the builder's fused RRF scale.
#
# Run: python examples/text_search.py   (after `maturin develop`)

from corvid import Db

CORPUS = [
    ("n1", "the quick brown fox jumps over the lazy dog"),
    ("n2", "a quick red fox leaps over a sleeping dog"),
    ("n3", "slow green turtle crosses the road"),
    ("n4", "东京是一座巨大的城市"),   # Tokyo is a huge city
    ("n5", "大阪是关西最大的城市"),   # Osaka is Kansai's biggest city
    ("n6", "机器学习正在改变数据库"),  # ML is changing databases
]

with Db.open_memory() as db:
    notes = db.collection("notes")
    for key, body in CORPUS:
        notes.insert(key, {"body": body})
    notes.create_text_index("body")

    def search(query, label):
        rows = notes.query().text("body", query, 3).run()
        hits = " ".join(f"{r.key}({r.score:.6f})" for r in rows)
        print(f"{label:<28} -> {hits}")

    def phrase(query, label):
        rows = notes.phrase_search("body", query, 3)
        hits = " ".join(f"{r.key}({r.score:.6f})" for r in rows)
        print(f"{label:<28} -> {hits}")

    search("quick fox", 'bm25 "quick fox":')
    search("quick dog", 'bm25 "quick dog":')
    search("城市", "bm25 CJK 城市 (city):")
    search("数据库", "bm25 CJK 数据库 (database):")

    phrase("fox jumps over", 'phrase "fox jumps over":')
    phrase("over jumps fox", "phrase reversed (no match):")
    phrase("leaps over a sleeping", "phrase stop words collapsed:")

    notes.close()
