"""Non-LLM next-POI baselines. Each returns a ranked list of POI ids (best first)."""
from __future__ import annotations

from collections import Counter
from datetime import datetime

from .retrieval import DEFAULTS, poi_seq, retrieve

# Time-aware heuristic settings, selected on the validation split by `python -m tracex_ir.tune_heuristic`.
HEURISTIC = DEFAULTS.get("heuristic", {"half_life_days": 90.0, "hour_w": 1.0, "weekday_w": 1.0, "hour_window": 1})


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


def _ts(c):
    return datetime.fromisoformat(c["time"])


def frequency_baseline(inst, history, pop, n=20, include_prefix=True, params=None):
    """The brief's heuristic: past visits counted by POI, hour and weekday, with recency decay.

    score(poi) = sum over the user's earlier check-ins at that POI of
        0.5 ** (age_days / half_life) * (1 + hour_w * [hour within window] + weekday_w * [same weekday])
    `age_days` is measured from the target's time. The ONLY target fields read are `time`,
    `hour` and `weekday` (the same fields the LLM prompts are given); never its POI or category.
    """
    P = {**HEURISTIC, **(params or {})}
    tgt = inst["target"]
    t0, hour, wd = _ts(tgt), tgt["hour"], tgt["weekday"]
    checkins = [c for t in history for c in t["checkins"]]
    if include_prefix:
        checkins += list(inst["prefix"])
    scores = Counter()
    for c in checkins:
        age = max((t0 - _ts(c)).total_seconds() / 86400.0, 0.0)
        d = abs(c["hour"] - hour)
        near = min(d, 24 - d) <= P["hour_window"]
        w = 0.5 ** (age / P["half_life_days"]) * (1 + P["hour_w"] * near + P["weekday_w"] * (c["weekday"] == wd))
        scores[c["poi_id"]] += w
    return _fill(scores, pop, n)


def plain_frequency_baseline(inst, history, pop, n=20, include_prefix=True):
    """Plain visit counter (no time or recency); kept as the simplest reference."""
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
