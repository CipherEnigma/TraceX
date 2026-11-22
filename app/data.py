"""Data access for the dashboard. Reuses tracex_llm.evaluate so every number matches the paper tables.

Caches are keyed by a signature of the results folder (file names, sizes, mtimes), so new
results from Kaggle show up on the next rerun without restarting the app.
"""
from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from tracex_data.api import load_dataset
from tracex_ir.baselines import GlobalPopularity, frequency_baseline
from tracex_ir.retrieval import IDF
from tracex_llm import config as C
from tracex_llm.evaluate import evaluate, frequency_rows, summarize

SAMPLES = {"sample_test_500": "sample_test_500", "sample_test_debug20": "sample_test_debug20"}
PLAN_MODELS = ("qwen7b", "qwen3b")
PLAN_METHODS = ("llmmob", "retrieval-jaccard")


def signature() -> tuple:
    if not C.RESULTS_DIR.exists():
        return ()
    return tuple(sorted((str(p.relative_to(C.RESULTS_DIR)), p.stat().st_size, int(p.stat().st_mtime))
                        for p in C.RESULTS_DIR.rglob("*") if p.is_file()))


def last_updated(sig: tuple) -> str:
    if not sig:
        return "no results yet"
    ts = max(s[2] for s in sig)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")  # local time


def has_results(sample: str, sig: tuple) -> bool:
    return any(name.endswith(f"__{sample}.jsonl") and "__mock__" not in name for name, _, _ in sig)


@st.cache_data(show_spinner=False)
def evaluation(sample: str, sig: tuple) -> dict:
    return evaluate(C.RESULTS_DIR, sample)


@st.cache_data(show_spinner=False)
def expected_n(dataset: str, sample: str) -> int:
    return len(load_dataset(dataset).instances(sample))


@st.cache_data(show_spinner=False)
def frequency_summary(dataset: str, sample: str) -> dict:
    return summarize(frequency_rows(dataset, sample), expected_n(dataset, sample))


def freq(res: dict, dataset: str, sample: str) -> dict:
    return res["frequency"].get(dataset) or frequency_summary(dataset, sample)


# ---------------------------------------------------------------------------
# The planned runs and their status
# ---------------------------------------------------------------------------
def method_parts(tag: str):
    return ("retrieval", tag.split("-", 1)[1]) if tag.startswith("retrieval-") else (tag, None)


def plan() -> list[dict]:
    out = []
    for d in ("nyc", "tky", "msteps_tokyo"):
        for method in PLAN_METHODS:
            for model in PLAN_MODELS:
                for cond in C.name_conditions(d):
                    out.append({"dataset": d, "method": method, "model": model, "condition": cond})
    return out


def run_key(r: dict, sample: str) -> str:
    m, retr = method_parts(r["method"])
    return C.run_name(r["dataset"], r["model"], m, r["condition"], retr, sample)


@st.cache_data(show_spinner=False)
def run_meta(sample: str, sig: tuple) -> dict:
    """Per run: gpu, sec/instance (weighted over sessions), last update, fingerprint."""
    out = {}
    for p in C.RESULTS_DIR.glob(f"*/*__{sample}.meta.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        sess = [s for s in m.get("sessions", []) if s.get("sec_per_instance") is not None]
        n = sum(s["todo"] for s in sess)
        spi = sum(s["sec_per_instance"] * s["todo"] for s in sess) / n if n else None
        last = m.get("sessions", [{}])[-1]
        when = last.get("finished") or last.get("started")
        out[m["run"]] = {"gpu": (last.get("backend") or {}).get("gpu"), "spi": spi, "fingerprint": m.get("fingerprint"),
                         "updated": when, "model": m["settings"]["model"]}
    return out


def fmt_when(iso: str | None) -> str:
    if not iso:
        return "–"
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%b %d %H:%M")
    except ValueError:
        return iso


# ---------------------------------------------------------------------------
# Instance-level data (explorer, E3 example)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def dataset(name: str):
    ds = load_dataset(name)
    train = ds.train_trajectories()
    pois = {int(r.poi_id): (r.category, (r["name"] if "name" in ds.pois.columns else None))
            for _, r in ds.pois.iterrows()}
    return {"ds": ds, "idf": IDF(train), "pop": GlobalPopularity(train), "pois": pois}


@st.cache_data(show_spinner=False)
def run_records(sample: str, sig: tuple) -> dict:
    """{run_key: {instance_id: record}} for all non-mock runs of the sample."""
    out = {}
    for p in C.RESULTS_DIR.glob(f"*/*__{sample}.jsonl"):
        if "__mock__" in p.name:
            continue
        recs = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[r["instance_id"]] = r
        out[p.stem] = recs
    return out


def frequency_ranking(name: str, inst: dict) -> list[int]:
    d = dataset(name)
    return frequency_baseline(inst, d["ds"].history(inst), d["pop"], n=C.N_PREDICTIONS)


# ---------------------------------------------------------------------------
# Personal-assistant reports (Members 1 and 2; formats in docs/assistant_plan.md, section 8)
# ---------------------------------------------------------------------------
def _assist_dir():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "reports" / "assistant"


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(open(path, encoding="utf-8").read())


def assistant_report(name: str) -> dict | None:
    """reports/assistant/<name>.json, re-read when the file changes; None if it does not exist yet."""
    p = _assist_dir() / f"{name}.json"
    if not p.exists():
        return None
    try:
        return _read_json(str(p), p.stat().st_mtime)
    except (OSError, ValueError):
        return None
