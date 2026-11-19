"""Comparison figures across the processed datasets (static PNGs for the report)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FIGURES_DIR, PROCESSED_DIR  # noqa: E402

# Categorical slots 1-3 of the validated reference palette, fixed per dataset.
COLORS = {"nyc": "#2a78d6", "tky": "#eb6834", "msteps_tokyo": "#1baf7a"}
LABELS = {"nyc": "Foursquare NYC", "tky": "Foursquare TKY", "msteps_tokyo": "Massive-STEPS Tokyo"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def _load(names):
    out = {}
    for n in names:
        p = PROCESSED_DIR / n / "checkins.csv"
        if p.exists():
            out[n] = pd.read_csv(p, parse_dates=["local_time"])
    return out


def make_figures(names=("nyc", "tky", "msteps_tokyo")) -> list[str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = _load(names)
    if not data:
        return []
    saved = []

    def save(fig, fname):
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        path = FIGURES_DIR / fname
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        saved.append(str(path))

    # 1. POI popularity long tail (rank vs train check-ins, log-log)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for n, df in data.items():
        f = df[df["split"] == "train"]["poi_id"].value_counts().values
        ax.plot(np.arange(1, len(f) + 1), f, color=COLORS[n], lw=2, label=LABELS[n])
    ax.set_xscale("log"); ax.set_yscale("log")
    _style(ax, "POI popularity is long-tailed", "POI rank (log)", "Training check-ins (log)")
    ax.legend(frameon=False)
    save(fig, "poi_popularity_longtail.png")

    # 2. Trajectory length distribution (share of trajectories, capped at 15+)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    width = 0.8 / len(data)
    for k, (n, df) in enumerate(data.items()):
        L = df.groupby("traj_id").size().clip(upper=15)
        share = L.value_counts(normalize=True).reindex(range(2, 16), fill_value=0)
        ax.bar(share.index + (k - (len(data) - 1) / 2) * width, share.values, width=width * 0.92,
               color=COLORS[n], label=LABELS[n])
    ax.set_xticks(range(2, 16), [str(i) if i < 15 else "15+" for i in range(2, 16)])
    _style(ax, "Trajectory length after 24-hour sessionisation", "Check-ins per trajectory",
           "Share of trajectories")
    ax.legend(frameon=False)
    save(fig, "trajectory_length.png")

    # 3. Hour-of-day profile
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for n, df in data.items():
        h = df["local_time"].dt.hour.value_counts(normalize=True).reindex(range(24), fill_value=0)
        ax.plot(h.index, h.values, color=COLORS[n], lw=2, marker="o", ms=3, label=LABELS[n])
    ax.set_xticks(range(0, 24, 3))
    _style(ax, "Check-ins by local hour", "Hour of day", "Share of check-ins")
    ax.legend(frameon=False)
    save(fig, "hour_of_day.png")

    # 4. Revisit vs new test targets (E2 split sizes)
    rows = []
    for n in data:
        st = json.loads((PROCESSED_DIR / n / "stats.json").read_text(encoding="utf-8"))
        t = st["instances"]["test"]
        rows.append((n, t["target_in_prefix_rate"], t["revisit_rate"] - t["target_in_prefix_rate"], t["new_poi_rate"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(rows))
    parts = [("In current prefix", "#2a78d6", 1), ("In history only", "#86b6ef", 2), ("New to user", "#c3c2b7", 3)]
    left = np.zeros(len(rows))
    for label, color, idx in parts:
        vals = np.array([r[idx] for r in rows])
        ax.barh(y, vals, left=left, color=color, label=label, edgecolor=SURFACE, linewidth=2, height=0.6)
        for yi, (l, v) in enumerate(zip(left, vals)):
            if v >= 0.06:
                ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if idx == 1 else INK)
        left += vals
    ax.set_yticks(y, [LABELS[r[0]] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    _style(ax, "Where test targets come from (all test instances)", "Share of test targets", "")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save(fig, "revisit_vs_new.png")
    return saved
