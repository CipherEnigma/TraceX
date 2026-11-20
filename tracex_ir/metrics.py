"""Ranking metrics. `ranked` is a list of ids, best first; `relevant` is a set of ids."""
from __future__ import annotations

import math


def precision_at_k(ranked, relevant, k):
    return sum(1 for r in ranked[:k] if r in relevant) / k


def recall_at_k(ranked, relevant, k):
    return sum(1 for r in ranked[:k] if r in relevant) / len(relevant) if relevant else 0.0


def average_precision(ranked, relevant):
    hits, total = 0, 0.0
    for i, r in enumerate(ranked, 1):
        if r in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked, relevant):
    for i, r in enumerate(ranked, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def accuracy_at_k(ranked, target, k):
    return float(target in ranked[:k])


def ndcg_at_k(ranked, relevant, k):
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(ranked[:k], 1) if r in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
