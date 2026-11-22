"""Select the time-aware heuristic's settings on the VALIDATION split only.

    python -m tracex_ir.tune_heuristic

Selection metric: mean val MRR across datasets. Result is merged into tracex_ir/defaults.json
under the key "heuristic" (the retrieval settings in that file are left untouched).
"""
from __future__ import annotations

import itertools
import json
import random

import numpy as np

from tracex_data.api import load_dataset

from . import metrics as M
from .baselines import GlobalPopularity, frequency_baseline
from .tune import DATASETS, DEFAULTS, N_VAL, SEED

GRID = {"half_life_days": [7.0, 30.0, 90.0, 365.0, 1e9], "hour_w": [0.0, 1.0, 2.0, 4.0],
        "weekday_w": [0.0, 0.5, 1.0, 2.0], "hour_window": [0, 1, 2]}


def main():
    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    table = {}
    for d in DATASETS:
        ds = load_dataset(d)
        pop = GlobalPopularity(ds.train_trajectories())
        insts = list(ds.instances("val"))
        random.Random(SEED).shuffle(insts)
        insts = [(ds.history(i), i) for i in insts[:N_VAL]]
        table[d] = []
        for cfg in configs:
            rr = [M.reciprocal_rank(frequency_baseline(i, h, pop, params=cfg), {i["target"]["poi_id"]}) for h, i in insts]
            table[d].append(float(np.mean(rr)))
        print(d, "done", flush=True)
    mean = np.mean([table[d] for d in DATASETS], axis=0)
    order = np.argsort(-mean)
    for j in order[:6]:
        print(f"  {configs[j]} mean MRR={mean[j]:.4f} " + " ".join(f"{d}={table[d][j]:.3f}" for d in DATASETS))
    best = configs[int(order[0])]
    plain = configs[[i for i, c in enumerate(configs) if c["hour_w"] == 0 and c["weekday_w"] == 0 and c["half_life_days"] == 1e9][0]]
    print("plain-count equivalent mean MRR:", round(float(mean[configs.index(plain)]), 4), "| selected:", best)
    out = json.loads(DEFAULTS.read_text(encoding="utf-8"))
    out["heuristic"] = {**best, "selected_on": "validation (500 instances per dataset, seed 42), mean MRR across datasets"}
    DEFAULTS.write_text(json.dumps(out, indent=2, default=list))


if __name__ == "__main__":
    main()
