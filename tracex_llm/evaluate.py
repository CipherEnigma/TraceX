"""Evaluate LLM runs for E1-E3 with paired statistics (Member 3).

    python -m tracex_llm.evaluate                         # all runs in results/llm/, sample_test_500
    python -m tracex_llm.evaluate --sample sample_test_debug20

Writes `reports/llm/evaluation.json` (everything) and `reports/llm/results.md` (paper tables).

Scoring (same for every method, on the same instances):
- Accuracy@1, Accuracy@5, MRR and NDCG@5 against the target POI. Rankings are cut to the
  N_PREDICTIONS (10) the LLM is asked for, so MRR is MRR@10 for every method.
- Behaviour rates: copy rate (top-1 is a POI from the user's history or current trajectory) and
  invalid rate (no usable top-1: unparseable, empty, or a POI id that does not exist).
- The frequency heuristic (Member 2's `frequency_baseline`) is recomputed here on exactly the same
  instances, so every LLM-vs-heuristic comparison is paired.
- 95% CIs by bootstrap over instances; differences by paired bootstrap; McNemar on Acc@1 and Acc@5;
  Holm correction within each experiment's family of tests.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tracex_data.api import load_dataset
from tracex_ir import metrics as M
from tracex_ir.baselines import GlobalPopularity, frequency_baseline

from . import config as C
from . import stats as S

METRICS = ("acc1", "acc5", "mrr", "ndcg5", "cat_acc1", "cat_acc5")
BINARY = ("acc1", "acc5", "cat_acc1", "cat_acc5")
FREQ = "frequency"


# ---------------------------------------------------------------------------
# Per-instance scores
# ---------------------------------------------------------------------------
def score(ranked: list, target: int, poi_to_cat: dict | None = None, target_cat: str | None = None) -> dict:
    r = ranked[:C.N_PREDICTIONS]
    s = {"acc1": M.accuracy_at_k(r, target, 1), "acc5": M.accuracy_at_k(r, target, 5),
         "mrr": M.reciprocal_rank(r, {target}), "ndcg5": M.ndcg_at_k(r, {target}, 5)}
    if poi_to_cat is not None and target_cat is not None:
        ranked_cats = [poi_to_cat.get(p) for p in r]
        s["cat_acc1"] = M.category_accuracy_at_k(ranked_cats, target_cat, 1)
        s["cat_acc5"] = M.category_accuracy_at_k(ranked_cats, target_cat, 5)
    else:
        s["cat_acc1"] = float("nan")
        s["cat_acc5"] = float("nan")
    return s


def method_tag(rec: dict) -> str:
    return f"retrieval-{rec['retriever']}" if rec["method"] == "retrieval" else rec["method"]


def _poi_to_cat(dataset: str) -> dict:
    ds = load_dataset(dataset)
    return dict(zip(ds.pois["poi_id"], ds.pois["category"]))


def _inst_target_cat(dataset: str, sample: str) -> dict:
    ds = load_dataset(dataset)
    return {inst["instance_id"]: inst["target"]["category"] for inst in ds.instances(sample)}


def load_runs(results_dir: Path, sample: str, include_mock: bool = False) -> dict:
    """{run_key: {"meta": {...}, "rows": {instance_id: row}}} for one sample."""
    runs = {}
    cat_maps: dict[str, dict] = {}
    target_cats: dict[str, dict] = {}
    for f in sorted(results_dir.glob(f"*/*__{sample}.jsonl")):
        rows, meta = {}, None
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["model"] == "mock" and not include_mock:
                break
            meta = meta or {k: r[k] for k in ("dataset", "model", "condition")} | {"method": method_tag(r)}
            ds_name = meta["dataset"]
            if ds_name not in cat_maps:
                cat_maps[ds_name] = _poi_to_cat(ds_name)
                target_cats[ds_name] = _inst_target_cat(ds_name, sample)
            p = r["parse"]
            tcat = target_cats[ds_name].get(r["instance_id"])
            rows[r["instance_id"]] = {
                **score(p["ranked"], r["target"], cat_maps[ds_name], tcat),
                "invalid": float(p["invalid_answer"]), "copy": float(bool(p["copied_top1"])),
                "copy_prompt": float(bool(p["copied_top1_prompt"])),
                "is_revisit": bool(r["is_revisit"]), "target_in_prompt": bool(r["target_in_prompt"]),
                "status": p["status"], "seconds": r.get("seconds"), "new_tokens": r.get("new_tokens"),
            }
        if rows:
            runs[f.stem] = {"meta": meta, "rows": rows}
    return runs


def frequency_rows(dataset: str, sample: str) -> dict:
    """Member 2's frequency heuristic, scored like the LLM runs on the same instances."""
    ds = load_dataset(dataset)
    pop = GlobalPopularity(ds.train_trajectories())
    poi_cat = _poi_to_cat(dataset)
    rows = {}
    for inst in ds.instances(sample):
        hist = ds.history(inst)
        ranked = frequency_baseline(inst, hist, pop, n=C.N_PREDICTIONS)
        inputs = {c["poi_id"] for t in hist for c in t["checkins"]} | {c["poi_id"] for c in inst["prefix"]}
        tcat = inst["target"]["category"]
        rows[inst["instance_id"]] = {**score(ranked, inst["target"]["poi_id"], poi_cat, tcat), "invalid": 0.0,
                                     "copy": float(ranked[0] in inputs), "copy_prompt": float("nan"),
                                     "is_revisit": bool(inst["is_revisit"]), "target_in_prompt": None,
                                     "status": "heuristic", "seconds": None, "new_tokens": None}
    return rows


# ---------------------------------------------------------------------------
# Summaries and comparisons
# ---------------------------------------------------------------------------
def summarize(rows: dict, expected_n: int | None = None, n_boot: int = S.N_BOOT) -> dict:
    vals = list(rows.values())
    out = {"n": len(vals), "complete": expected_n is None or len(vals) == expected_n}
    for m in METRICS + ("invalid", "copy"):
        out[m] = S.bootstrap_ci([v[m] for v in vals], n_boot)
    cp = [v["copy_prompt"] for v in vals if not np.isnan(v["copy_prompt"])]
    out["copy_prompt"] = float(np.mean(cp)) if cp else None
    out["status"] = dict(Counter(v["status"] for v in vals))
    secs = [v["seconds"] for v in vals if v["seconds"] is not None]
    out["sec_per_instance"] = float(np.mean(secs)) if secs else None
    toks = [v["new_tokens"] for v in vals if v["new_tokens"] is not None]
    out["mean_new_tokens"] = float(np.mean(toks)) if toks else None
    # E2: revisit vs new (target in the user's history or current trajectory, or not)
    for tag, flag in (("revisit", True), ("new", False)):
        sub = [v for v in vals if v["is_revisit"] == flag]
        out[tag] = {"n": len(sub), **{m: S.bootstrap_ci([v[m] for v in sub], n_boot)
                                      for m in ("acc1", "acc5", "mrr", "cat_acc1", "cat_acc5")}}
    return out


def compare(a: dict, b: dict, n_boot: int = S.N_BOOT, subset=None) -> dict:
    """Paired comparison of run a vs run b on their common instances (optionally a subset)."""
    ids = sorted(set(a) & set(b))
    if subset is not None:
        ids = [i for i in ids if subset(a[i])]
    out = {"n": len(ids)}
    for m in METRICS:
        out[m] = S.paired_bootstrap([a[i][m] for i in ids], [b[i][m] for i in ids], n_boot)
    for m in BINARY:
        out[f"mcnemar_{m}"] = S.mcnemar([a[i][m] for i in ids], [b[i][m] for i in ids])
    return out


def evaluate(results_dir: Path = C.RESULTS_DIR, sample: str = "sample_test_500",
             include_mock: bool = False, n_boot: int = S.N_BOOT) -> dict:
    runs = load_runs(results_dir, sample, include_mock)
    datasets = sorted({r["meta"]["dataset"] for r in runs.values()})
    expected = {d: len(load_dataset(d).instances(sample)) for d in datasets}
    freq = {d: frequency_rows(d, sample) for d in datasets}

    res = {"sample": sample, "n_boot": n_boot, "alpha": S.ALPHA, "runs": {}, "frequency": {},
           "e1": [], "e2": [], "e3": [], "models": []}
    for d in datasets:
        res["frequency"][d] = summarize(freq[d], expected[d], n_boot)
    for key, r in runs.items():
        res["runs"][key] = {**r["meta"], **summarize(r["rows"], expected[r["meta"]["dataset"]], n_boot)}

    # E1 (+ E2 per subset): every LLM run vs the frequency heuristic on the same instances.
    # On Massive-STEPS only the "present" condition enters E1 (the realistic setting).
    for key, r in runs.items():
        m = r["meta"]
        if m["condition"] not in (C.NO_NAMES, "present"):
            continue
        res["e1"].append({"run": key, **m, "vs_frequency": compare(r["rows"], freq[m["dataset"]], n_boot)})
        for tag, flag in (("revisit", True), ("new", False)):
            res["e2"].append({"run": key, **m, "subset": tag,
                              "vs_frequency": compare(r["rows"], freq[m["dataset"]], n_boot,
                                                      subset=lambda v, f=flag: v["is_revisit"] == f)})

    # E3: name conditions, same model and method, paired on the same instances.
    by = defaultdict(dict)
    for key, r in runs.items():
        m = r["meta"]
        if m["dataset"] in C.NAME_DATASETS:
            by[(m["dataset"], m["model"], m["method"])][m["condition"]] = key
    for (d, model, method), conds in sorted(by.items()):
        for a, b in (("present", "aliased"), ("present", "removed"), ("aliased", "removed")):
            if a in conds and b in conds:
                res["e3"].append({"dataset": d, "model": model, "method": method, "a": a, "b": b,
                                  **compare(runs[conds[a]]["rows"], runs[conds[b]]["rows"], n_boot)})

    # Size comparison: 7B vs 3B, same dataset / method / condition.
    pairs = defaultdict(dict)
    for key, r in runs.items():
        m = r["meta"]
        pairs[(m["dataset"], m["method"], m["condition"])][m["model"]] = key
    for (d, method, cond), models in sorted(pairs.items()):
        if "qwen7b" in models and "qwen3b" in models:
            res["models"].append({"dataset": d, "method": method, "condition": cond,
                                  **compare(runs[models["qwen7b"]]["rows"], runs[models["qwen3b"]]["rows"], n_boot)})

    # Holm correction on Acc@1 McNemar p-values within each family
    for fam, get in (("e1", lambda x: x["vs_frequency"]), ("e2", lambda x: x["vs_frequency"]),
                     ("e3", lambda x: x), ("models", lambda x: x)):
        adj = S.holm([get(x)["mcnemar_acc1"]["p"] for x in res[fam]])
        for x, p in zip(res[fam], adj):
            get(x)["mcnemar_acc1"]["p_holm"] = p
    return res


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------
def _ci(s, pct=True):
    f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.3f}")
    return f"{f(s['mean'])} [{f(s['lo'])}, {f(s['hi'])}]"


def _diff(s):
    return f"{100 * s['diff']:+.1f} [{100 * s['lo']:+.1f}, {100 * s['hi']:+.1f}]"


def _p(p):
    return "n/a" if p is None or np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def to_markdown(res: dict) -> str:
    L = [f"# TraceX LLM results ({res['sample']})", "",
         f"Values in %, 95% bootstrap CI in brackets ({res['n_boot']} resamples). Differences are paired "
         "over the same instances; p = McNemar on Acc@1, Holm-adjusted within each table. "
         "Copy = top-1 is a POI from the user's history or current trajectory. "
         "Invalid = no usable top-1 POI.", ""]
    L += ["## E1: heuristic floor", "",
          "| Dataset | Method | Model | n | Acc@1 | Acc@5 | Cat-Acc@1 | Cat-Acc@5 | MRR | NDCG@5 | Copy | Invalid | Δ Acc@1 vs freq | p |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d, f in sorted(res["frequency"].items()):
        L.append(f"| {d} | frequency | - | {f['n']} | {_ci(f['acc1'])} | {_ci(f['acc5'])} | "
                 f"{_ci(f['cat_acc1'])} | {_ci(f['cat_acc5'])} | "
                 f"{_ci(f['mrr'], False)} | {_ci(f['ndcg5'], False)} | {100 * f['copy']['mean']:.1f} | 0.0 | - | - |")
        for x in sorted((x for x in res["e1"] if x["dataset"] == d), key=lambda x: (x["model"], x["method"])):
            r, v = res["runs"][x["run"]], x["vs_frequency"]
            flag = "" if r["complete"] else " (incomplete)"
            L.append(f"| {d} | {x['method']} | {x['model']} | {r['n']}{flag} | {_ci(r['acc1'])} | {_ci(r['acc5'])} | "
                     f"{_ci(r['cat_acc1'])} | {_ci(r['cat_acc5'])} | "
                     f"{_ci(r['mrr'], False)} | {_ci(r['ndcg5'], False)} | {100 * r['copy']['mean']:.1f} | "
                     f"{100 * r['invalid']['mean']:.1f} | {_diff(v['acc1'])} | {_p(v['mcnemar_acc1']['p_holm'])} |")
    L += ["", "## E2: revisit vs new targets (Acc@1)", "",
          "Revisit = the target POI appears in the user's history or current trajectory (fixed definition, "
          "docs/data_pipeline.md).", "",
          "| Dataset | Method | Model | Subset | n | LLM Acc@1 | Freq Acc@1 | LLM Cat-Acc@1 | Freq Cat-Acc@1 | Δ vs freq | p |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(res["e2"], key=lambda x: (x["dataset"], x["model"], x["method"], x["subset"] != "revisit")):
        r, f, v = res["runs"][x["run"]], res["frequency"][x["dataset"]], x["vs_frequency"]
        L.append(f"| {x['dataset']} | {x['method']} | {x['model']} | {x['subset']} | {r[x['subset']]['n']} | "
                 f"{_ci(r[x['subset']]['acc1'])} | {_ci(f[x['subset']]['acc1'])} | "
                 f"{_ci(r[x['subset']]['cat_acc1'])} | {_ci(f[x['subset']]['cat_acc1'])} | {_diff(v['acc1'])} | "
                 f"{_p(v['mcnemar_acc1']['p_holm'])} |")
    if res["e3"]:
        L += ["", "## E3: venue-name test (Massive-STEPS)", "",
              "| Model | Method | Condition | n | Acc@1 | Acc@5 | MRR | Copy | Invalid |", "|---|---|---|---|---|---|---|---|---|"]
        for key, r in sorted(res["runs"].items()):
            if r["dataset"] in C.NAME_DATASETS:
                L.append(f"| {r['model']} | {r['method']} | {r['condition']} | {r['n']} | {_ci(r['acc1'])} | "
                         f"{_ci(r['acc5'])} | {_ci(r['mrr'], False)} | {100 * r['copy']['mean']:.1f} | "
                         f"{100 * r['invalid']['mean']:.1f} |")
        L += ["", "| Model | Method | Comparison | n | Δ Acc@1 | Δ Acc@5 | p (Acc@1) |", "|---|---|---|---|---|---|---|"]
        for x in res["e3"]:
            L.append(f"| {x['model']} | {x['method']} | {x['a']} − {x['b']} | {x['n']} | {_diff(x['acc1'])} | "
                     f"{_diff(x['acc5'])} | {_p(x['mcnemar_acc1']['p_holm'])} |")
    if res["models"]:
        L += ["", "## Model size: 7B − 3B", "", "| Dataset | Method | Condition | n | Δ Acc@1 | Δ Acc@5 | p |",
              "|---|---|---|---|---|---|---|"]
        for x in res["models"]:
            L.append(f"| {x['dataset']} | {x['method']} | {x['condition']} | {x['n']} | {_diff(x['acc1'])} | "
                     f"{_diff(x['acc5'])} | {_p(x['mcnemar_acc1']['p_holm'])} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tracex_llm.evaluate")
    ap.add_argument("--sample", default="sample_test_500")
    ap.add_argument("--results-dir", type=Path, default=C.RESULTS_DIR)
    ap.add_argument("--out-dir", type=Path, default=C.LLM_REPORTS_DIR)
    ap.add_argument("--n-boot", type=int, default=S.N_BOOT)
    ap.add_argument("--include-mock", action="store_true", help="also evaluate mock runs (testing only)")
    a = ap.parse_args(argv)
    res = evaluate(a.results_dir, a.sample, a.include_mock, a.n_boot)
    if not res["runs"]:
        raise SystemExit(f"no LLM results for {a.sample} in {a.results_dir}")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if a.sample == "sample_test_500" else f"_{a.sample}"
    (a.out_dir / f"evaluation{suffix}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = to_markdown(res)
    (a.out_dir / f"results{suffix}.md").write_text(md, encoding="utf-8")
    try:
        print(md)
    except UnicodeEncodeError:  # Windows consoles (cp1252) cannot show the Delta sign
        print(md.encode("ascii", "replace").decode("ascii"))
    print(f"[evaluate] wrote {a.out_dir}")
    return res


if __name__ == "__main__":
    main()
