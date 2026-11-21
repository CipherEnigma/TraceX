"""Paper figures for E1-E3 from `reports/llm/evaluation.json` (Member 3).

    python -m tracex_llm.evaluate      # first: writes evaluation.json and results.md
    python -m tracex_llm.report        # then: reports/llm/figures/*.png and *.pdf

Figures (all bars show the mean with its 95% bootstrap CI; every number is also in results.md):
- e1_heuristic_floor : Acc@1 / Acc@5 of each LLM run, with the frequency heuristic as a dashed floor.
- e2_revisit_new     : Acc@1 on revisit vs new targets for the heuristic and each LLM run.
- e3_name_test       : Acc@1 / Acc@5 with venue names present, removed and aliased (Massive-STEPS).
- behaviour          : copy rate and invalid-answer rate per run.

Colour: the reference categorical palette's first three slots (validated colour-blind safe for
all pairs). Identity never relies on colour alone: every figure has a legend, and E3 bars carry labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402

FIG_DIR = C.LLM_REPORTS_DIR / "figures"

# Reference palette (light), slots 1-3, plus ink and surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

DATASET = {"nyc": "Foursquare NYC", "tky": "Foursquare TKY", "msteps_tokyo": "Massive-STEPS Tokyo"}
MODEL = {"qwen7b": "Qwen2.5-7B", "qwen3b": "Qwen2.5-3B", "mock": "mock"}
MODEL_ORDER = ["qwen7b", "qwen3b", "mock"]
CONDITION = {"present": "Names present", "removed": "Names removed", "aliased": "Names aliased"}


def method_label(m: str) -> str:
    if m == "llmmob":
        return "LLM-Mob"
    if m.startswith("retrieval-"):
        return f"Retrieval ({m.split('-', 1)[1].upper() if m.endswith('lcs') else m.split('-', 1)[1].title()})"
    return m


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "legend.fontsize": 8.5,
    })


def _pct(s):
    """mean, lower err, upper err in percent from a {'mean','lo','hi'} summary."""
    return 100 * s["mean"], 100 * (s["mean"] - s["lo"]), 100 * (s["hi"] - s["mean"])


def _bars(ax, x, summaries, color, width, label=None, hatch=None):
    vals = [_pct(s) for s in summaries]
    m = [v[0] for v in vals]
    err = np.array([[v[1] for v in vals], [v[2] for v in vals]])
    ax.bar(x, m, width, color=color, edgecolor=SURFACE, linewidth=1.5, label=label, hatch=hatch, zorder=2)
    ax.errorbar(x, m, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2.5, zorder=3)
    return m


def _save(fig, name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _main_runs(res):
    """LLM runs in the realistic setting (no names, or names present on Massive-STEPS)."""
    return [r for r in res["runs"].values() if r["condition"] in (C.NO_NAMES, "present")]


def _models(runs):
    present = {r["model"] for r in runs}
    return [m for m in MODEL_ORDER if m in present]


# ---------------------------------------------------------------------------
# E1
# ---------------------------------------------------------------------------
def fig_e1(res, out_dir):
    runs = _main_runs(res)
    datasets = [d for d in DATASET if d in res["frequency"]]
    models = _models(runs)
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.3 * len(datasets), 5.4), sharey="row", squeeze=False)
    w = 0.8 / max(1, len(models))
    for j, d in enumerate(datasets):
        methods = sorted({r["method"] for r in runs if r["dataset"] == d})
        for i, metric in enumerate(("acc1", "acc5")):
            ax = axes[i, j]
            f = res["frequency"][d][metric]
            ax.axhspan(100 * f["lo"], 100 * f["hi"], color=MUTED, alpha=0.15, lw=0, zorder=1)
            ax.axhline(100 * f["mean"], color=INK_2, ls="--", lw=1.2, zorder=1,
                       label="Frequency heuristic (95% CI)")
            for k, model in enumerate(models):
                summ, xs = [], []
                for mi, meth in enumerate(methods):
                    r = next((r for r in runs if r["dataset"] == d and r["method"] == meth
                              and r["model"] == model), None)
                    if r:
                        summ.append(r[metric])
                        xs.append(mi - 0.4 + w * (k + 0.5))
                if summ:
                    _bars(ax, xs, summ, SERIES[k], w * 0.92, label=MODEL[model])
            ax.set_xticks(range(len(methods)), [method_label(m) for m in methods])
            ax.set_xlim(-0.6, len(methods) - 0.4)
            if i == 0:
                ax.set_title(DATASET[d])
            if j == 0:
                ax.set_ylabel(f"{'Accuracy@1' if metric == 'acc1' else 'Accuracy@5'} (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("E1 · LLM accuracy against the frequency heuristic", y=1.07, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "e1_heuristic_floor", out_dir)


# ---------------------------------------------------------------------------
# E2
# ---------------------------------------------------------------------------
def _short_method(m: str) -> str:
    return "LLM-Mob" if m == "llmmob" else ("Retrieval" if m.startswith("retrieval") else m)


def fig_e2(res, out_dir):
    runs = _main_runs(res)
    datasets = [d for d in DATASET if d in res["frequency"]]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.8), sharey=True, squeeze=False)
    for j, d in enumerate(datasets):
        ax = axes[0, j]
        entries = [("Frequency\nheuristic", res["frequency"][d])]
        for model in _models(runs):
            for r in sorted((r for r in runs if r["dataset"] == d and r["model"] == model), key=lambda r: r["method"]):
                entries.append((f"{_short_method(r['method'])}\n{MODEL[model].split('-')[-1]}", r))
        x = np.arange(len(entries))
        for k, tag in enumerate(("revisit", "new")):
            summ = [e[1][tag]["acc1"] for e in entries]
            means = _bars(ax, x - 0.2 + 0.4 * k, summ, SERIES[k], 0.37,
                          label="Revisit target" if tag == "revisit" else "New target")
            if tag == "new":  # the new-target values are the E2 finding; label them (they are often ~0)
                for x_, m_, s_ in zip(x + 0.2, means, summ):
                    ax.text(x_, 100 * s_["hi"] + 0.6, f"{m_:.1f}", ha="center", va="bottom", fontsize=7,
                            color=INK_2)
        n_rev, n_new = res["frequency"][d]["revisit"]["n"], res["frequency"][d]["new"]["n"]
        ax.set_title(f"{DATASET[d]}\n(revisit n={n_rev}, new n={n_new})")
        ax.set_xticks(x, [e[0] for e in entries], fontsize=8)
        if j == 0:
            ax.set_ylabel("Accuracy@1 (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("E2 · Accuracy on revisited vs new POIs", y=1.12, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "e2_revisit_new", out_dir)


# ---------------------------------------------------------------------------
# E3
# ---------------------------------------------------------------------------
def fig_e3(res, out_dir):
    runs = [r for r in res["runs"].values() if r["dataset"] in C.NAME_DATASETS]
    if not runs:
        return []
    models = _models(runs)
    conds = [c for c in C.NAME_CONDITIONS if any(r["condition"] == c for r in runs)]
    fig, axes = plt.subplots(2, len(models), figsize=(3.8 * len(models), 5.6), sharey="row", squeeze=False)
    w = 0.8 / len(conds)
    for j, model in enumerate(models):
        methods = sorted({r["method"] for r in runs if r["model"] == model})
        for i, metric in enumerate(("acc1", "acc5")):
            ax = axes[i, j]
            for k, c in enumerate(conds):
                xs, summ = [], []
                for mi, meth in enumerate(methods):
                    r = next((r for r in runs if r["model"] == model and r["method"] == meth
                              and r["condition"] == c), None)
                    if r:
                        xs.append(mi - 0.4 + w * (k + 0.5))
                        summ.append(r[metric])
                if not summ:
                    continue
                means = _bars(ax, xs, summ, SERIES[k], w * 0.92, label=CONDITION[c])
                for x_, m_, s_ in zip(xs, means, summ):  # direct labels (relief for the low-contrast slot)
                    ax.text(x_, 100 * s_["hi"] + 0.8, f"{m_:.1f}", ha="center", va="bottom",
                            fontsize=7, color=INK_2)
            ax.set_xticks(range(len(methods)), [method_label(m) for m in methods])
            ax.set_xlim(-0.6, len(methods) - 0.4)
            ax.margins(y=0.12)
            if i == 0:
                ax.set_title(MODEL[model])
            if j == 0:
                ax.set_ylabel(f"{'Accuracy@1' if metric == 'acc1' else 'Accuracy@5'} (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.02))
    n = runs[0]["n"]
    fig.suptitle(f"E3 · Venue-name test on Massive-STEPS Tokyo (n={n}, same instances in every condition)",
                 y=1.07, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "e3_name_test", out_dir)


# ---------------------------------------------------------------------------
# Behaviour rates
# ---------------------------------------------------------------------------
def fig_behaviour(res, out_dir):
    runs = list(res["runs"].values())
    if not runs:
        return []
    cond_order = {C.NO_NAMES: 0, "present": 1, "removed": 2, "aliased": 3}
    settings = sorted({(r["dataset"], r["method"], r["condition"]) for r in runs},
                      key=lambda s: (list(DATASET).index(s[0]), s[1], cond_order.get(s[2], 9)))
    labels = [f"{DATASET[d]} · {_short_method(m)}" + (f" · names {c}" if c != C.NO_NAMES else "")
              for d, m, c in settings]
    models = _models(runs)
    h = 0.8 / len(models)
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(settings) + 1.5), sharey=True)
    y = np.arange(len(settings))
    for ax, metric, title in ((axes[0], "copy", "Copy rate\n(top-1 POI is from the user's own history)"),
                              (axes[1], "invalid", "Invalid-answer rate\n(no usable top-1 POI)")):
        for k, model in enumerate(models):
            ys, vals = [], []
            for i, s in enumerate(settings):
                r = next((r for r in runs if (r["dataset"], r["method"], r["condition"]) == s
                          and r["model"] == model), None)
                if r:
                    ys.append(i - 0.4 + h * (k + 0.5))
                    vals.append(_pct(r[metric]))
            ax.barh(ys, [v[0] for v in vals], h * 0.9, color=SERIES[k], edgecolor=SURFACE,
                    linewidth=1.5, label=MODEL[model], zorder=2)
            ax.errorbar([v[0] for v in vals], ys, xerr=np.array([[v[1] for v in vals], [v[2] for v in vals]]),
                        fmt="none", ecolor=INK_2, elinewidth=1, capsize=2, zorder=3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("% of instances")
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 100)
    axes[0].set_yticks(y, labels, fontsize=7.5)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    return _save(fig, "behaviour", out_dir)


def make_figures(res: dict, out_dir: Path = FIG_DIR) -> list[Path]:
    _style()
    paths = []
    for f in (fig_e1, fig_e2, fig_e3, fig_behaviour):
        paths += f(res, out_dir)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tracex_llm.report")
    ap.add_argument("--eval", type=Path, default=C.LLM_REPORTS_DIR / "evaluation.json",
                    help="output of `python -m tracex_llm.evaluate`")
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    a = ap.parse_args(argv)
    if not a.eval.exists():
        raise SystemExit(f"{a.eval} not found - run `python -m tracex_llm.evaluate` first")
    res = json.loads(a.eval.read_text(encoding="utf-8"))
    for p in make_figures(res, a.out_dir):
        print(f"[report] {p}")


if __name__ == "__main__":
    main()
