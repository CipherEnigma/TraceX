"""Choose retrieval settings on the VALIDATION split only (never the test split).

    python -m tracex_ir.tune

Selection metric: retrieval MAP over validation instances that have at least one relevant
history trajectory (relevant = contains the target POI; target used only for scoring).
Writes tracex_ir/defaults.json, which the rest of the code and Member 3 read.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import numpy as np

from tracex_data.api import load_dataset

from . import metrics as M
from .retrieval import IDF, poi_seq, retrieve

DEFAULTS = Path(__file__).with_name("defaults.json")
DATASETS = ["nyc", "tky", "msteps_tokyo"]
N_VAL = 500
SEED = 42

CONFIGS = {"jaccard": {}, "lcs": {}, "tfidf": {}}
for k1, b in itertools.product([0.5, 1.2, 1.5, 2.0], [0.0, 0.25, 0.5, 0.75, 1.0]):
    CONFIGS[f"bm25(k1={k1},b={b})"] = {"method": "bm25", "k1": k1, "b": b}
for combo in [("bm25", "lcs"), ("bm25", "jaccard"), ("bm25", "tfidf"), ("bm25", "lcs", "jaccard")]:
    CONFIGS["hybrid(" + "+".join(combo) + ")"] = {"method": "hybrid", "methods": combo}


def val_map(ds, insts, idf, method, kw):
    scores = []
    for inst in insts:
        hist, tgt = ds.history(inst), inst["target"]["poi_id"]
        rel = {t["traj_id"] for t in hist if tgt in poi_seq(t)}
        if not rel:
            continue
        query = [c["poi_id"] for c in inst["prefix"]]
        ranked = [t["traj_id"] for t, _ in retrieve(method, query, hist, idf, **kw)]
        scores.append(M.average_precision(ranked, rel))
    return float(np.mean(scores))


def main():
    table = {}
    for d in DATASETS:
        ds = load_dataset(d)
        idf = IDF(ds.train_trajectories())
        insts = list(ds.instances("val"))
        random.Random(SEED).shuffle(insts)
        insts = insts[:N_VAL]
        table[d] = {}
        for name, cfg in CONFIGS.items():
            kw = dict(cfg)
            method = kw.pop("method", name)
            table[d][name] = val_map(ds, insts, idf, method, kw)
        print(f"{d}: val MAP of recent control = {val_map(ds, insts, idf, 'recent', {}):.3f}")
    names = list(CONFIGS)
    mean_map = {n: float(np.mean([table[d][n] for d in DATASETS])) for n in names}
    for n in sorted(names, key=lambda n: -mean_map[n])[:8]:
        print(f"  {n:34s} mean={mean_map[n]:.4f} " + " ".join(f"{d}={table[d][n]:.3f}" for d in DATASETS))
    best = max(names, key=lambda n: mean_map[n])
    cfg = dict(CONFIGS[best])
    method = cfg.pop("method", best)
    out = {"method": method, "params": cfg, "top_k": 3, "selected_on": "validation (500 instances per dataset, seed 42)",
           "selection_metric": "mean retrieval MAP across datasets", "val_map": table, "val_mean_map": mean_map}
    DEFAULTS.write_text(json.dumps(out, indent=2, default=list))
    print("selected:", best)


if __name__ == "__main__":
    main()
