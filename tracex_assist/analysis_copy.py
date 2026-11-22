"""What does the LLM copy, and where does it beat the counter? (Member 1, task 1.4)

    python -m tracex_assist.analysis_copy      # every saved run -> reports/assistant/copy_analysis.{json,md}

A saved top-1 guess is compared with simple rules computed from the same input the model had: the time-aware
heuristic, the plain visit counter, the last place visited, the user's Markov top-1 and the current trip. The target
is used only for scoring; the rules read nothing of it but its time, as the prompts did.

Layout of the JSON (for the dashboard):
    datasets[d]   n, truth{rule}, random_place{rule}, markov_coverage, counters{validation, test}
    runs[key]     dataset, model, method, condition, main, n, complete, acc1, new_hits,
                  match{rule}, excess{rule}, outside{other_own, never_seen, invalid}, ties{...}, disagree{...},
                  vs{comparator}{slice}: n, llm, comparator, diff, lo, hi, llm_only, comparator_only, p, p_holm, beats
All rates are fractions. `main` marks the runs of E1; the others are the name variants of E3.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tracex_data.api import load_dataset
from tracex_data.config import DATASETS, REPORTS_DIR
from tracex_ir.baselines import GlobalPopularity, frequency_baseline, plain_frequency_baseline
from tracex_llm import config as C
from tracex_llm import stats as S
from tracex_llm.evaluate import method_tag

from .profile import time_bucket

OUT_DIR = REPORTS_DIR / "assistant"

RULES = {"heuristic": "Time-aware heuristic", "counter": "Visit counter", "last": "Last place",
         "markov": "Markov", "trip": "Current trip", "any_rule": "Any of these"}
COMPARATORS = {"heuristic": "Time-aware heuristic", "counter": "Visit counter",
               "heuristic_history": "Heuristic, history only"}
HISTORY_BINS = (("1-9", 9), ("10-40", 40), ("41-150", 150), ("151+", float("inf")))  # LLM-Mob shows the last 40
SLICES = {
    "history": [label for label, _ in HISTORY_BINS],
    "time": ["morning", "midday", "afternoon", "evening", "night"],
    "kind": ["revisit", "new"],
}


# ---------------------------------------------------------------------------
# The rules, per instance
# ---------------------------------------------------------------------------
def _most_common(counts: Counter) -> tuple[int | None, set[int]]:
    """Top item (ties by POI id, as in the baselines) and every item tied with it."""
    if not counts:
        return None, set()
    best = max(counts.values())
    tied = {p for p, c in counts.items() if c == best}
    return min(tied), tied


def rule_guesses(inst: dict, history: list[dict], pop: GlobalPopularity) -> dict:
    """What each rule would answer. Reads only the prefix, the history and the target's time."""
    trips = [t["checkins"] for t in history] + [inst["prefix"]]
    last = inst["prefix"][-1]["poi_id"]
    visits = Counter(c["poi_id"] for t in trips for c in t)
    following = Counter(b["poi_id"] for t in trips for a, b in zip(t, t[1:]) if a["poi_id"] == last)
    markov, markov_tied = _most_common(following)  # no fallback: None if the user never moved on from here
    return {
        "heuristic": frequency_baseline(inst, history, pop, n=1)[0],
        "counter": plain_frequency_baseline(inst, history, pop, n=1)[0],
        "heuristic_history": frequency_baseline(inst, history, pop, n=1, include_prefix=False)[0],
        "last": last,
        "markov": markov,
        "trip": {c["poi_id"] for c in inst["prefix"]},
        "counter_tied": _most_common(visits)[1],
        "markov_tied": markov_tied,
        "own": set(visits),
    }


def matches(poi: int | None, g: dict) -> dict[str, bool]:
    """Which rules would have given this POI."""
    out = {r: poi is not None and poi == g[r] for r in ("heuristic", "counter", "last", "markov")}
    out["trip"] = poi in g["trip"]
    out["any_rule"] = any(out.values())
    return out


def history_bin(n_checkins: int) -> str:
    return next(label for label, upper in HISTORY_BINS if n_checkins <= upper)


def instance_slices(inst: dict) -> set[str]:
    return {"all", history_bin(inst["n_history_checkins"]), time_bucket(inst["target"]["hour"]),
            "revisit" if inst["is_revisit"] else "new"}


# ---------------------------------------------------------------------------
# Datasets and runs
# ---------------------------------------------------------------------------
def counters_on_validation(ds, pop: GlobalPopularity) -> dict:
    """The comparators on the validation split, so that the history-only variant is not picked on test."""
    insts = ds.instances("val")
    hits = Counter()
    for inst in insts:
        g = rule_guesses(inst, ds.history(inst), pop)
        for c in COMPARATORS:
            hits[c] += g[c] == inst["target"]["poi_id"]
    return {"n": len(insts), **{c: hits[c] / len(insts) if insts else None for c in COMPARATORS}}


def analyse_dataset(name: str, sample: str, n_boot: int) -> tuple[dict, dict]:
    """(summary, {instance_id: the instance with its rule guesses})."""
    ds = load_dataset(name)
    pop = GlobalPopularity(ds.train_trajectories())
    cases = {}
    for inst in ds.instances(sample):
        g = rule_guesses(inst, ds.history(inst), pop)
        cases[inst["instance_id"]] = {"inst": inst, "g": g, "slices": instance_slices(inst),
                                      "truth": matches(inst["target"]["poi_id"], g)}
    items = list(cases.values())
    gs = [x["g"] for x in items]

    def chance(rule: str, g: dict) -> float:
        """Probability that one of the user's places, drawn at random, matches the rule."""
        if rule in ("trip", "any_rule"):
            chosen = set(g["trip"])
            if rule == "any_rule":
                chosen |= {g["heuristic"], g["counter"], g["last"], g["markov"]}
            return len(chosen & g["own"]) / len(g["own"])
        return (g[rule] in g["own"]) / len(g["own"])

    summary = {
        "n": len(items),
        "truth": {r: S.bootstrap_ci([x["truth"][r] for x in items], n_boot) for r in RULES},
        "random_place": {r: float(np.mean([chance(r, g) for g in gs])) for r in RULES},
        "markov_coverage": float(np.mean([g["markov"] is not None for g in gs])),
        "counters": {
            "validation": counters_on_validation(ds, pop),
            "test": {c: {"acc1": float(np.mean([x["g"][c] == x["inst"]["target"]["poi_id"] for x in items])),
                         "equals_last": float(np.mean([g[c] == g["last"] for g in gs]))} for c in COMPARATORS},
        },
    }
    return summary, cases


def load_guesses(results_dir: Path, sample: str) -> dict:
    """{run key: {meta, guesses {instance_id: record}}}; mock runs and truncated lines are skipped."""
    runs = {}
    for f in sorted(results_dir.glob(f"*/*__{sample}.jsonl")):
        rows, meta = {}, None
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # blank, or the last line of an interrupted run
                continue
            if r["model"] == "mock":
                break
            meta = meta or {"dataset": r["dataset"], "model": r["model"], "method": method_tag(r),
                            "condition": r["condition"]}
            rows[r["instance_id"]] = r
        if rows:
            runs[f.stem] = {"meta": meta, "guesses": rows}
    return runs


def analyse_run(meta: dict, records: dict, cases: dict, n_boot: int) -> dict:
    unknown = set(records) - set(cases)
    moved = [i for i, r in records.items() if i in cases and r["target"] != cases[i]["inst"]["target"]["poi_id"]]
    if unknown or moved:
        raise ValueError(f"{meta}: saved run does not fit the processed data "
                         f"({len(unknown)} unknown instances, {len(moved)} different targets)")

    ids = sorted(records)
    rows = [cases[i] for i in ids]
    # an unusable top-1 (empty, unparsable, or not a POI) matches nothing
    top = [None if records[i]["parse"]["invalid_answer"] else records[i]["parse"]["ranked"][0] for i in ids]
    found = [matches(p, x["g"]) for p, x in zip(top, rows)]
    right = np.array([p == x["inst"]["target"]["poi_id"] for p, x in zip(top, rows)])

    differ = [(p, x["g"]) for p, x in zip(top, rows) if x["g"]["counter"] != x["g"]["last"]]
    to_counter = sum(p == g["counter"] for p, g in differ)
    to_last = sum(p == g["last"] for p, g in differ)

    out = {
        **meta,
        "main": meta["condition"] in (C.NO_NAMES, "present"),
        "n": len(ids),
        "complete": len(ids) == len(cases),
        "acc1": float(right.mean()),
        "new_hits": {"hits": sum(bool(r) for r, x in zip(right, rows) if not x["inst"]["is_revisit"]),
                     "n": sum(not x["inst"]["is_revisit"] for x in rows)},
        "match": {r: S.bootstrap_ci([f[r] for f in found], n_boot) for r in RULES},
        "excess": {r: S.paired_bootstrap([f[r] for f in found], [x["truth"][r] for x in rows], n_boot)
                   for r in RULES},
        "outside": {
            "other_own": float(np.mean([not f["any_rule"] and p in x["g"]["own"]
                                        for f, p, x in zip(found, top, rows)])),
            "never_seen": float(np.mean([p is not None and p not in x["g"]["own"] for p, x in zip(top, rows)])),
            "invalid": float(np.mean([p is None for p in top])),
        },
        "ties": {r: float(np.mean([p in x["g"][f"{r}_tied"] for p, x in zip(top, rows)]))
                 for r in ("counter", "markov")},
        "disagree": {"n": len(differ),
                     **{k: v / len(differ) if differ else None
                        for k, v in (("counter", to_counter), ("last", to_last),
                                     ("neither", len(differ) - to_counter - to_last))}},
        "vs": {},
    }

    for comp in COMPARATORS:
        theirs = np.array([x["g"][comp] == x["inst"]["target"]["poi_id"] for x in rows])
        out["vs"][comp] = {}
        for name in ["all"] + [s for group in SLICES.values() for s in group]:
            mask = np.array([name in x["slices"] for x in rows])
            if not mask.any():
                continue
            a, b = right[mask], theirs[mask]
            diff, mc = S.paired_bootstrap(a, b, n_boot), S.mcnemar(a, b)
            out["vs"][comp][name] = {
                "n": int(mask.sum()), "llm": float(a.mean()), "comparator": float(b.mean()),
                "diff": diff["diff"], "lo": diff["lo"], "hi": diff["hi"],
                "llm_only": mc["a_only"], "comparator_only": mc["b_only"], "p": mc["p"]}
    return out


def adjust(runs: dict) -> None:
    """Holm per comparator and grouping, E1 runs and name variants apart (the families of tracex_llm.evaluate)."""
    group_of = {s: group for group, names in SLICES.items() for s in names}
    families = defaultdict(list)
    for r in runs.values():
        for comp, cells in r["vs"].items():
            for name, cell in cells.items():
                families[comp, r["main"], group_of.get(name, name)].append(cell)
    for cells in families.values():
        for cell, p in zip(cells, S.holm([c["p"] for c in cells])):
            cell["p_holm"] = p
            cell["beats"] = bool(cell["diff"] > 0 and p < S.ALPHA)


def analyse(results_dir: Path = C.RESULTS_DIR, sample: str = "sample_test_500", n_boot: int = S.N_BOOT) -> dict:
    saved = load_guesses(results_dir, sample)
    if not saved:
        raise FileNotFoundError(f"no LLM results for {sample} in {results_dir}")
    res = {"sample": sample, "n_boot": n_boot, "alpha": S.ALPHA, "rules": RULES, "comparators": COMPARATORS,
           "slices": SLICES, "datasets": {}, "runs": {}}
    cases = {}
    order = list(DATASETS)
    found = {r["meta"]["dataset"] for r in saved.values()}
    for d in sorted(found, key=lambda d: (order.index(d) if d in order else len(order), d)):
        res["datasets"][d], cases[d] = analyse_dataset(d, sample, n_boot)
    for key, r in saved.items():
        res["runs"][key] = analyse_run(r["meta"], r["guesses"], cases[r["meta"]["dataset"]], n_boot)
    adjust(res["runs"])
    return res


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def _delta(c: dict | None) -> str:
    if not c:
        return "-"
    return f"{100 * c['diff']:+.1f}{'*' if c['p_holm'] < S.ALPHA else ''} (n={c['n']})"


def _table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(str(x) for x in row) + " |" for row in rows), ""]


def _label(r: dict) -> str:
    cond = "" if r["condition"] == C.NO_NAMES else f", {r['condition']}"
    return f"{r['model']} {r['method']}{cond}" + ("" if r["complete"] else " (incomplete)")


def to_markdown(res: dict) -> str:
    data = res["datasets"]
    runs = sorted(res["runs"].values(), key=lambda r: (list(data).index(r["dataset"]), not r["main"], r["model"],
                                                       r["method"], r["condition"]))
    main_runs = [r for r in runs if r["main"]]

    L = [f"# What does the LLM copy? ({res['sample']})", "",
         "Generated by `python -m tracex_assist.analysis_copy`; do not edit by hand. Values in %. Intervals and "
         "p-values are in `copy_analysis.json`.", "",
         "- **Time-aware heuristic:** the project's frequency baseline (hour and recency, current trip included).",
         "- **Visit counter:** plain visit counts over history and current trip.",
         "- **Last place:** the last check-in before the one to predict.",
         "- **Markov:** where this user went most often after the last place; no match if they never left it.",
         "- **Current trip:** any place already in the trip.", "",
         "The rules overlap, so a row does not add up to 100. Matching a rule shows that the output is the same as "
         "the rule's, not how the model arrived at it.", "",
         "## 1. The LLM's first guess equals ...", "",
         "*Actual next place* is how often the true answer follows the rule. *Random own place* is what guessing "
         "uniformly among the user's places would match.", ""]
    for d, v in data.items():
        rows = [["*Actual next place*"] + [_pct(v["truth"][r]["mean"]) for r in RULES] + ["-"] * 3,
                ["*Random own place*"] + [_pct(v["random_place"][r]) for r in RULES] + ["-"] * 3]
        rows += [[_label(r)] + [_pct(r["match"][k]["mean"]) for k in RULES] + [_pct(x) for x in r["outside"].values()]
                 for r in runs if r["dataset"] == d]
        L += [f"### {d} (n = {v['n']}; Markov defined for {_pct(v['markov_coverage'])}%)", ""]
        L += _table(["Guess"] + list(RULES.values()) + ["Other own place", "Never seen", "Invalid"], rows)

    L += ["## 2. Copying beyond what is true", "",
          "How much more often the guess follows a rule than the true next place does, in points [95% CI].", ""]
    L += _table(["Dataset", "Run"] + list(RULES.values()),
                [[r["dataset"], _label(r)] + [f"{100 * x['diff']:+.1f} [{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}]"
                                             for x in (r["excess"][k] for k in RULES)] for r in main_runs])

    L += ["## 3. When the visit counter and the last place disagree, whom does the LLM follow?", ""]
    L += _table(["Dataset", "Run", "n", "Counter", "Last place", "Neither"],
                [[r["dataset"], _label(r), r["disagree"]["n"]]
                 + [_pct(r["disagree"][k]) for k in ("counter", "last", "neither")] for r in runs])

    L += ["Ties: with equal counts the rules pick the lowest POI id. Counting any of the tied places as a match:", ""]
    L += _table(["Dataset", "Run", "Counter", "Counter, any tied", "Markov", "Markov, any tied"],
                [[r["dataset"], _label(r), _pct(r["match"]["counter"]["mean"]), _pct(r["ties"]["counter"]),
                  _pct(r["match"]["markov"]["mean"]), _pct(r["ties"]["markov"])] for r in main_runs])

    L += ["## 4. Does the LLM beat the counter?", "",
          "Δ Acc@1 = LLM - counter, paired on the same instances. * = significant either way (McNemar, "
          "Holm-corrected per counter and grouping; E1 runs and name variants are separate families, as in the "
          "E1 and E2 tables).", ""]
    L += _table(["Dataset", "Run", "LLM Acc@1"] + [f"Δ vs {x}" for x in COMPARATORS.values()],
                [[r["dataset"], _label(r), _pct(r["acc1"])] + [_delta(r["vs"][c]["all"]) for c in COMPARATORS]
                 for r in runs])

    L += ["### The counters themselves", "",
          "*History only* is the same heuristic without the current trip. It is an **exploratory** comparator, added "
          "after seeing how often the heuristic answers with the place the user is already at. The validation "
          "column shows whether it would also have been chosen without looking at test.", ""]
    L += _table(["Dataset", "Counter", "Validation Acc@1", "Test Acc@1", "Top-1 is the current place"],
                [[d, label, _pct(v["counters"]["validation"][c]), _pct(v["counters"]["test"][c]["acc1"]),
                  _pct(v["counters"]["test"][c]["equals_last"])]
                 for d, v in data.items() for c, label in COMPARATORS.items()])

    L += ["## 5. Where does the LLM beat the counter?", "",
          "History = the user's check-ins in the training history. Time = bucket of the check-in to predict.", ""]
    titles = {"history": "history length", "time": "time of day", "kind": "revisit or new"}
    for comp, label in COMPARATORS.items():
        for group, names in SLICES.items():
            L += [f"### Δ Acc@1 vs {label}, by {titles[group]}", ""]
            L += _table(["Dataset", "Run"] + names,
                        [[r["dataset"], _label(r)] + [_delta(r["vs"][comp].get(s)) for s in names]
                         for r in main_runs])

    L += ["### Subgroups where the LLM is significantly better", ""]
    wins = [[r["dataset"], _label(r), COMPARATORS[c], s, _delta(x)] for r in main_runs
            for c, cells in r["vs"].items() for s, x in cells.items() if s != "all" and x["beats"]]
    L += _table(["Dataset", "Run", "Counter", "Subgroup", "Δ"], wins) if wins else ["None.", ""]

    hits, n = sum(r["new_hits"]["hits"] for r in main_runs), sum(r["new_hits"]["n"] for r in main_runs)
    L += [f"On never-visited targets the LLMs got {hits} of {n} first guesses right (E1 runs)."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tracex_assist.analysis_copy")
    ap.add_argument("--sample", default="sample_test_500")
    ap.add_argument("--results-dir", type=Path, default=C.RESULTS_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=S.N_BOOT)
    a = ap.parse_args(argv)
    res = analyse(a.results_dir, a.sample, a.n_boot)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if a.sample == "sample_test_500" else f"_{a.sample}"
    text = {"json": json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False) + "\n", "md": to_markdown(res)}
    for ext, body in text.items():
        (a.out_dir / f"copy_analysis{suffix}.{ext}").write_text(body, encoding="utf-8", newline="\n")
    print(f"[analysis_copy] {len(res['runs'])} runs -> {a.out_dir}")
    return res


if __name__ == "__main__":
    main()
