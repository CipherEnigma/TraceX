"""Retrieval of similar historical trajectories (Jaccard, LCS, TF-IDF, BM25, hybrid).

Query = the current trajectory prefix (POI ids). Candidates = `ds.history(inst)`.
Every retriever returns `[(trajectory, score), ...]`, best first. Ties are broken by
recency (more recent trajectory first). The target is never used here.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


def load_defaults():
    """Settings chosen on the validation split by `python -m tracex_ir.tune`."""
    f = Path(__file__).with_name("defaults.json")
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {"method": "bm25", "params": {"k1": 1.5, "b": 0.75}, "top_k": 3}


DEFAULTS = load_defaults()
_BM25 = DEFAULTS["params"] if DEFAULTS["method"] == "bm25" else {}


def poi_seq(traj):
    return [c["poi_id"] for c in traj["checkins"]]


def _rank(cands, scores):
    # candidates are oldest-first, so reverse index = recency tie-break
    order = sorted(range(len(cands)), key=lambda i: (-scores[i], -i))
    return [(cands[i], scores[i]) for i in order]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def lcs_length(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def lcs_similarity(a, b):
    """LCS length normalised by the longer sequence (in [0, 1])."""
    return lcs_length(a, b) / max(len(a), len(b)) if a and b else 0.0


class IDF:
    """Document frequencies over a corpus of trajectories (use train trajectories)."""

    def __init__(self, trajectories):
        self.n = 0
        self.df = Counter()
        lens = 0
        for t in trajectories:
            seq = poi_seq(t)
            self.n += 1
            lens += len(seq)
            self.df.update(set(seq))
        self.avgdl = lens / self.n if self.n else 1.0

    def tfidf_idf(self, term):  # smooth idf, always > 0
        return math.log((1 + self.n) / (1 + self.df.get(term, 0))) + 1

    def bm25_idf(self, term):  # Lucene-style, always >= 0
        d = self.df.get(term, 0)
        return math.log(1 + (self.n - d + 0.5) / (d + 0.5))


def tfidf_cosine(query, doc, idf: IDF):
    q, d = Counter(query), Counter(doc)
    vq = {t: c * idf.tfidf_idf(t) for t, c in q.items()}
    vd = {t: c * idf.tfidf_idf(t) for t, c in d.items()}
    dot = sum(w * vd[t] for t, w in vq.items() if t in vd)
    nq = math.sqrt(sum(w * w for w in vq.values()))
    nd = math.sqrt(sum(w * w for w in vd.values()))
    return dot / (nq * nd) if nq and nd else 0.0


def bm25(query, doc, idf: IDF, k1=_BM25.get("k1", 1.5), b=_BM25.get("b", 0.75)):
    tf = Counter(doc)
    norm = k1 * (1 - b + b * len(doc) / idf.avgdl)
    return sum(idf.bm25_idf(t) * tf[t] * (k1 + 1) / (tf[t] + norm) for t in set(query) if t in tf)


def retrieve(method, query, cands, idf: IDF | None = None, **kw):
    """Rank candidate trajectories for a query POI sequence with one method."""
    if method == "recent":  # no-similarity control: most recent first
        return _rank(cands, [0.0] * len(cands))
    seqs = [poi_seq(c) for c in cands]
    if method == "jaccard":
        s = [jaccard(query, x) for x in seqs]
    elif method == "lcs":
        s = [lcs_similarity(query, x) for x in seqs]
    elif method == "tfidf":
        s = [tfidf_cosine(query, x, idf) for x in seqs]
    elif method == "bm25":
        s = [bm25(query, x, idf, **kw) for x in seqs]
    elif method == "hybrid":
        return hybrid(query, cands, idf, **kw)
    else:
        raise ValueError(f"unknown method {method}")
    return _rank(cands, s)


def hybrid(query, cands, idf, methods=("bm25", "lcs", "jaccard"), k=60):
    """Reciprocal rank fusion of several retrievers."""
    fused = [0.0] * len(cands)
    index = {id(c): i for i, c in enumerate(cands)}
    for m in methods:
        for rank, (c, _) in enumerate(retrieve(m, query, cands, idf), 1):
            fused[index[id(c)]] += 1.0 / (k + rank)
    return _rank(cands, fused)


METHODS = ["recent", "jaccard", "lcs", "tfidf", "bm25", "hybrid"]


def to_demonstration(traj, score=None):
    """Format shared with Member 3: one retrieved trajectory as a prompt demonstration.

    The last check-in is the 'answer' of the demonstration, the rest is its context.
    """
    ch = traj["checkins"]
    return {"traj_id": traj["traj_id"], "score": score,
            "context": ch[:-1], "next": ch[-1], "checkins": ch}


def retrieve_demonstrations(inst, history, idf, method=None, top_k=None):
    """Top-k demonstrations for an instance (query = inst['prefix'] only).

    Defaults (method, top_k) are the validation-selected ones in defaults.json.
    """
    method = method or DEFAULTS["method"]
    top_k = top_k or DEFAULTS["top_k"]
    query = [c["poi_id"] for c in inst["prefix"]]
    return [to_demonstration(t, s) for t, s in retrieve(method, query, history, idf)[:top_k]]
