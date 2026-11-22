"""Offline evaluation of the evidence engine (Member 2, task 2.5).

    python -m tracex_assist.evaluate
    python -m tracex_assist.evaluate --dataset nyc --sample sample_test_500

Measures Acc@1/5, MRR, NDCG@5 and category-level Acc@1/5 for:
    frequency   - plain visit counter (from tracex_ir.baselines)
    markov      - bigram: last place -> next place
    recency     - most recent places first
    engine      - evidence engine's fused ranking (gather().habits)

Plus an exploratory LLM fusion: the saved LLM ranking fused with the engine via RRF. Clearly labelled
post hoc since the LLM was not trained on these labels.

All baselines and the engine respect the leakage rule: they see the user's history (training
trajectories), the current-day prefix and the target's time/weekday, but never the target place.
Category accuracy uses `category_accuracy_at_k` from tracex_ir.metrics.

Writes `reports/assistant/results.json` and `results.md`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tracex_data.api import load_dataset
from tracex_ir import metrics as M
from tracex_llm import stats as S

from .contract import Context
from .evidence import gather
from .stub import build_profile, place_index

DATASETS = ["nyc", "tky", "msteps_tokyo"]
SAMPLE = "sample_test_500"
METRICS = ("acc1", "acc5", "mrr", "ndcg5", "cat_acc1", "cat_acc5")
BINARY = ("acc1", "acc5", "cat_acc1", "cat_acc5")
OUT = Path(__file__).resolve().parent.parent / "reports" / "assistant"


def _poi_to_cat(dataset: str) -> dict:
    ds = load_dataset(dataset)
    return dict(zip(ds.pois["poi_id"], ds.pois["category"]))


def _score(ranked: list[int], target: int, poi_cat: dict, target_cat: str) -> dict:
    r = ranked[:10]
    ranked_cats = [poi_cat.get(p) for p in r]
    return {
        "acc1": M.accuracy_at_k(r, target, 1),
        "acc5": M.accuracy_at_k(r, target, 5),
        "mrr": M.reciprocal_rank(r, {target}),
        "ndcg5": M.ndcg_at_k(r, {target}, 5),
        "cat_acc1": M.category_accuracy_at_k(ranked_cats, target_cat, 1),
        "cat_acc5": M.category_accuracy_at_k(ranked_cats, target_cat, 5),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _frequency_ranking(profile) -> list[int]:
    return [pid for pid, _ in profile.top_places]


def _markov_ranking(profile, current_poi: int | None) -> list[int]:
    if current_poi is None:
        return []
    after: Counter = Counter()
    for trip in profile.trips:
        for a, b in zip(trip, trip[1:]):
            if a.poi_id == current_poi:
                after[b.poi_id] += 1
    return [pid for pid, _ in after.most_common()]


def _recency_ranking(profile) -> list[int]:
    seen, result = set(), []
    for trip in reversed(profile.trips):
        for v in reversed(trip):
            if v.poi_id not in seen:
                seen.add(v.poi_id)
                result.append(v.poi_id)
    return result


def _engine_ranking(profile, context, places) -> list[int]:
    ev = gather(profile, context, places)
    return [s.poi_id for s in ev.habits]


# ---------------------------------------------------------------------------
# LLM fusion (exploratory, post hoc)
# ---------------------------------------------------------------------------
def _load_llm_rankings(dataset: str, sample: str) -> dict[str, list[int]]:
    """Load saved LLM top-10 from results/llm/. Returns {instance_id: ranked_pois}."""
    results_dir = Path(__file__).resolve().parent.parent / "results" / "llm" / dataset
    best = {}
    for f in sorted(results_dir.glob(f"*__{sample}.jsonl")):
        if "qwen7b" not in f.stem:
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("model") == "mock":
                continue
            best[r["instance_id"]] = r["parse"]["ranked"]
    return best


def _rrf_two(a: list[int], b: list[int], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in [a, b]:
        for rank, pid in enumerate(ranking, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(datasets=None, sample: str = SAMPLE, n_boot: int = S.N_BOOT) -> dict:
    datasets = datasets or DATASETS
    res = {"sample": sample, "n_boot": n_boot, "methods": {}, "engine": {}, "comparisons": []}

    for ds_name in datasets:
        ds = load_dataset(ds_name)
        places = place_index(ds_name)
        poi_cat = _poi_to_cat(ds_name)
        llm_ranks = _load_llm_rankings(ds_name, sample)
        insts = ds.instances(sample)

        methods = {
            "frequency": {},
            "markov": {},
            "recency": {},
            "engine": {},
        }
        if llm_ranks:
            methods["llm_fusion"] = {}

        for inst in insts:
            iid = inst["instance_id"]
            target = inst["target"]["poi_id"]
            target_cat = inst["target"]["category"]
            history = ds.history(inst)
            profile = build_profile(ds_name, str(inst["user_id"]), history)
            current_poi = inst["prefix"][-1]["poi_id"] if inst["prefix"] else None
            context = Context(weekday=inst["target"]["weekday"], hour=inst["target"]["hour"],
                              current_poi=current_poi)

            rankings = {
                "frequency": _frequency_ranking(profile),
                "markov": _markov_ranking(profile, current_poi),
                "recency": _recency_ranking(profile),
                "engine": _engine_ranking(profile, context, places),
            }
            if llm_ranks and iid in llm_ranks:
                rankings["llm_fusion"] = _rrf_two(rankings["engine"], llm_ranks[iid])

            for name, ranked in rankings.items():
                if name in methods:
                    methods[name][iid] = {
                        **_score(ranked, target, poi_cat, target_cat),
                        "is_revisit": bool(inst["is_revisit"]),
                    }

        for name, rows in methods.items():
            vals = list(rows.values())
            summary = {"dataset": ds_name, "n": len(vals)}
            for m in METRICS:
                summary[m] = S.bootstrap_ci([v[m] for v in vals], n_boot)
            for tag, flag in (("revisit", True), ("new", False)):
                sub = [v for v in vals if v["is_revisit"] == flag]
                summary[tag] = {"n": len(sub),
                                **{m: S.bootstrap_ci([v[m] for v in sub], n_boot)
                                   for m in ("acc1", "acc5", "cat_acc1", "cat_acc5")}}
            res["methods"].setdefault(name, {})[ds_name] = summary

        # Store engine summary at top level for the confidence dict
        if "engine" in methods:
            eng = res["methods"]["engine"][ds_name]
            res["engine"][ds_name] = {"acc1": eng["acc1"]["mean"], "acc5": eng["acc5"]["mean"],
                                      "n": eng["n"]}

        # Paired comparisons: each method vs frequency
        freq_rows = methods["frequency"]
        for name in ["markov", "recency", "engine"] + (["llm_fusion"] if "llm_fusion" in methods else []):
            other_rows = methods[name]
            ids = sorted(set(freq_rows) & set(other_rows))
            comp = {"dataset": ds_name, "method": name, "n": len(ids)}
            for m in METRICS:
                comp[m] = S.paired_bootstrap([other_rows[i][m] for i in ids],
                                              [freq_rows[i][m] for i in ids], n_boot)
            for m in BINARY:
                comp[f"mcnemar_{m}"] = S.mcnemar([other_rows[i][m] for i in ids],
                                                   [freq_rows[i][m] for i in ids])
            res["comparisons"].append(comp)

    # Holm correction on acc1 McNemar within all comparisons
    adj = S.holm([c["mcnemar_acc1"]["p"] for c in res["comparisons"]])
    for c, p in zip(res["comparisons"], adj):
        c["mcnemar_acc1"]["p_holm"] = p

    return res


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _ci(s, pct=True):
    f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.3f}")
    return f"{f(s['mean'])} [{f(s['lo'])}, {f(s['hi'])}]"


def _diff(s):
    return f"{100 * s['diff']:+.1f} [{100 * s['lo']:+.1f}, {100 * s['hi']:+.1f}]"


def _p(p):
    return "n/a" if p is None or (isinstance(p, float) and np.isnan(p)) else (
        "<0.001" if p < 0.001 else f"{p:.3f}")


def to_markdown(res: dict) -> str:
    L = [f"# TraceX assistant baselines ({res['sample']})", "",
         f"95% bootstrap CI ({res['n_boot']} resamples). "
         "Comparisons are paired; p = McNemar on Acc@1, Holm-adjusted.", ""]

    L += ["## Overall", "",
          "| Dataset | Method | n | Acc@1 | Acc@5 | Cat-Acc@1 | Cat-Acc@5 | MRR | NDCG@5 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ds_name in DATASETS:
        for name in ["frequency", "markov", "recency", "engine", "llm_fusion"]:
            if name not in res["methods"] or ds_name not in res["methods"][name]:
                continue
            s = res["methods"][name][ds_name]
            L.append(f"| {ds_name} | {name} | {s['n']} | {_ci(s['acc1'])} | {_ci(s['acc5'])} | "
                     f"{_ci(s['cat_acc1'])} | {_ci(s['cat_acc5'])} | "
                     f"{_ci(s['mrr'], False)} | {_ci(s['ndcg5'], False)} |")

    L += ["", "## Revisit vs new (Acc@1, Cat-Acc@1)", "",
          "| Dataset | Method | Subset | n | Acc@1 | Cat-Acc@1 |",
          "|---|---|---|---|---|---|"]
    for ds_name in DATASETS:
        for name in ["frequency", "markov", "recency", "engine", "llm_fusion"]:
            if name not in res["methods"] or ds_name not in res["methods"][name]:
                continue
            s = res["methods"][name][ds_name]
            for tag in ("revisit", "new"):
                sub = s[tag]
                L.append(f"| {ds_name} | {name} | {tag} | {sub['n']} | "
                         f"{_ci(sub['acc1'])} | {_ci(sub['cat_acc1'])} |")

    L += ["", "## vs frequency baseline", "",
          "| Dataset | Method | n | Δ Acc@1 | Δ Cat-Acc@1 | p (Acc@1) |",
          "|---|---|---|---|---|---|"]
    for c in res["comparisons"]:
        L.append(f"| {c['dataset']} | {c['method']} | {c['n']} | {_diff(c['acc1'])} | "
                 f"{_diff(c['cat_acc1'])} | {_p(c['mcnemar_acc1']['p_holm'])} |")

    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tracex_assist.evaluate")
    ap.add_argument("--dataset", nargs="+", default=DATASETS)
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--n-boot", type=int, default=S.N_BOOT)
    a = ap.parse_args(argv)
    res = evaluate(a.dataset, a.sample, a.n_boot)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = to_markdown(res)
    (OUT / "results.md").write_text(md, encoding="utf-8")
    try:
        print(md)
    except UnicodeEncodeError:
        print(md.encode("ascii", "replace").decode("ascii"))
    print(f"\n[evaluate] wrote {OUT}")
    return res


if __name__ == "__main__":
    main()
