"""Evaluate retrieval quality and non-LLM baselines.

    python -m tracex_ir.evaluate --dataset nyc --sample sample_test_500 --ks 1 3 5

Retrieval quality: a candidate (history trajectory) is *relevant* if it contains the target POI
(target used only for scoring). Instances with no relevant candidate are excluded from
Precision/Recall/MAP/MRR and reported as `coverage`.
Next-POI prediction: Accuracy@1/5, MRR, NDCG@5 on the full instance set, plus a revisit/new split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tracex_data.api import load_dataset

from . import metrics as M
from .baselines import GlobalPopularity, frequency_baseline, plain_frequency_baseline, retrieval_baseline
from .retrieval import METHODS, IDF, poi_seq, retrieve

OUT = Path(__file__).resolve().parent.parent / "reports" / "retrieval"


def evaluate(dataset, sample="sample_test_500", ks=(1, 3, 5), top_k=3, methods=METHODS):
    ds = load_dataset(dataset)
    train = ds.train_trajectories()
    idf, pop = IDF(train), GlobalPopularity(train)
    insts = ds.instances(sample)
    retr = {}
    for m in methods:
        retr[m] = {f"{n}@{k}": [] for n in ("P", "R", "F1") for k in ks}
        retr[m].update({"MAP": [], "MRR": []})
    pred = {name: {"acc1": [], "acc5": [], "mrr": [], "ndcg5": [], "revisit": []}
            for name in ["frequency", "plain_frequency"] + [f"{m}_vote" for m in methods]}
    n_rel = 0
    for inst in insts:
        hist, tgt = ds.history(inst), inst["target"]["poi_id"]
        query = [c["poi_id"] for c in inst["prefix"]]
        rel_ids = {t["traj_id"] for t in hist if tgt in poi_seq(t)}
        n_rel += bool(rel_ids)
        if rel_ids:
            for m in methods:
                ranked = [t["traj_id"] for t, _ in retrieve(m, query, hist, idf)]
                r = retr[m]
                for k in ks:
                    r[f"P@{k}"].append(M.precision_at_k(ranked, rel_ids, k))
                    r[f"R@{k}"].append(M.recall_at_k(ranked, rel_ids, k))
                    r[f"F1@{k}"].append(M.f1_at_k(ranked, rel_ids, k))
                r["MAP"].append(M.average_precision(ranked, rel_ids))
                r["MRR"].append(M.reciprocal_rank(ranked, rel_ids))
        outs = {"frequency": frequency_baseline(inst, hist, pop),
                "plain_frequency": plain_frequency_baseline(inst, hist, pop)}
        for m in methods:
            outs[f"{m}_vote"] = retrieval_baseline(m, inst, hist, idf, pop, top_k)
        for name, ranked in outs.items():
            p = pred[name]
            p["acc1"].append(M.accuracy_at_k(ranked, tgt, 1))
            p["acc5"].append(M.accuracy_at_k(ranked, tgt, 5))
            p["mrr"].append(M.reciprocal_rank(ranked, {tgt}))
            p["ndcg5"].append(M.ndcg_at_k(ranked, {tgt}, 5))
            p["revisit"].append(bool(inst["is_revisit"]))

    def mean(x):
        return float(np.mean(x)) if len(x) else float("nan")

    res = {"dataset": dataset, "sample": sample, "n_instances": len(insts),
           "coverage": n_rel / len(insts), "retrieval": {}, "prediction": {}}
    for m, r in retr.items():
        res["retrieval"][m] = {k: mean(v) for k, v in r.items()}
    for name, p in pred.items():
        rv = np.array(p["revisit"])
        row = {k: mean(p[k]) for k in ("acc1", "acc5", "mrr", "ndcg5")}
        for tag, mask in (("revisit", rv), ("new", ~rv)):
            row[f"acc1_{tag}"] = mean(np.array(p["acc1"])[mask])
            row[f"acc5_{tag}"] = mean(np.array(p["acc5"])[mask])
        res["prediction"][name] = row
    return res


def plot(res, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    tag, w = res["dataset"], 0.2
    ret = res["retrieval"]
    names = list(ret)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, key in enumerate(["P@1", "R@5", "MAP", "MRR"]):
        ax.bar(np.arange(len(names)) + i * w, [ret[n][key] for n in names], w, label=key)
    ax.set_xticks(np.arange(len(names)) + 1.5 * w, names)
    ax.set_title(f"Retrieval quality - {tag} (coverage {res['coverage']:.0%})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_retrieval.png", dpi=150)
    plt.close(fig)
    pr = res["prediction"]
    names = list(pr)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, key in enumerate(["acc1", "acc5", "mrr", "ndcg5"]):
        ax.bar(np.arange(len(names)) + i * w, [pr[n][key] for n in names], w, label=key)
    ax.set_xticks(np.arange(len(names)) + 1.5 * w, names, rotation=30, ha="right")
    ax.set_title(f"Non-LLM next-POI baselines - {tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_baselines.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["nyc", "tky", "msteps_tokyo"])
    ap.add_argument("--sample", default="sample_test_500")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--top-k", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in a.dataset:
        res = evaluate(d, a.sample, tuple(a.ks), a.top_k)
        (OUT / f"{d}_results.json").write_text(json.dumps(res, indent=2))
        plot(res, OUT)
        print(f"== {d}: {res['n_instances']} instances, retrieval coverage {res['coverage']:.1%}")
        for m, r in res["retrieval"].items():
            print(f"  {m:8s} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()))
        for n, r in res["prediction"].items():
            print(f"  {n:14s} acc1={r['acc1']:.3f} acc5={r['acc5']:.3f} mrr={r['mrr']:.3f} ndcg5={r['ndcg5']:.3f}")


if __name__ == "__main__":
    main()
