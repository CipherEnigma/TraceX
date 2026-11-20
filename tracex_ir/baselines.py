"""Non-LLM next-POI baselines. Each returns a ranked list of POI ids (best first)."""
from __future__ import annotations

from collections import Counter

from .retrieval import poi_seq, retrieve


class GlobalPopularity:
    def __init__(self, train_trajectories):
        c = Counter()
        for t in train_trajectories:
            c.update(poi_seq(t))
        self.ranking = [p for p, _ in c.most_common()]


def _fill(scored: Counter, pop: GlobalPopularity, n):
    ranked = [p for p, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))]
    seen = set(ranked)
    for p in pop.ranking:
        if len(ranked) >= n:
            break
        if p not in seen:
            ranked.append(p)
    return ranked[:n]


def frequency_baseline(inst, history, pop, n=20, include_prefix=True):
    """Rank POIs by how often this user visited them (history + current prefix)."""
    c = Counter()
    for t in history:
        c.update(poi_seq(t))
    if include_prefix:
        c.update(c_["poi_id"] for c_ in inst["prefix"])
    return _fill(c, pop, n)


def retrieval_baseline(method, inst, history, idf, pop, top_k=3, n=20):
    """Rank POIs by similarity-weighted votes from the top-k retrieved trajectories.

    A POI that follows the matched part of the query gets no special treatment;
    this is a plain bag-of-POIs vote, so it isolates retrieval quality.
    """
    query = [c["poi_id"] for c in inst["prefix"]]
    votes = Counter()
    for t, s in retrieve(method, query, history, idf)[:top_k]:
        w = s if s > 0 else 1e-6
        for p in poi_seq(t):
            votes[p] += w
    return _fill(votes, pop, n)
