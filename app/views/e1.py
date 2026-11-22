"""Page 2: E1 heuristic floor - LLM runs against the frequency heuristic."""
from __future__ import annotations

import streamlit as st

import context
import data
import ui
from tracex_llm import config as C

METRICS = {"acc1": "Acc@1", "acc5": "Acc@5", "mrr": "MRR", "ndcg5": "NDCG@5"}
METRIC_LONG = {"acc1": "Accuracy@1", "acc5": "Accuracy@5", "mrr": "MRR", "ndcg5": "NDCG@5"}
METHOD_ORDER = ["llmmob", "retrieval-jaccard", "retrieval-lcs", "retrieval-bm25"]


def _run_cmd(dataset: str) -> str:
    return (f"# on Kaggle: STAGE = \"e1\"  (or locally on a GPU)\n"
            f"python -m tracex_llm.run --dataset {dataset} --model qwen7b --method llmmob retrieval")


def _dataset_block(dataset: str, metric: str, models: list[str], methods: list[str], res: dict, sample: str):
    f = data.freq(res, dataset, sample)
    e1 = {x["run"]: x for x in res["e1"] if x["dataset"] == dataset}
    runs = [(k, res["runs"][k]) for k in e1 if res["runs"][k]["model"] in models and res["runs"][k]["method"] in methods]
    exp = data.expected_n(dataset, sample)

    ui.section(f"{METRIC_LONG[metric]} by method · {ui.DATASETS[dataset]}",
               f"n = {exp} per bar unless tagged partial")
    if not runs:
        ui.empty_state("No LLM results yet for this selection",
                       f"Frequency heuristic for {ui.DATASETS[dataset]}: {ui.pct(f['acc1'])} Acc@1, n = {f['n']}. "
                       "LLM bars appear here as soon as a run's results are copied into results/llm/.",
                       _run_cmd(dataset), status_html=ui.status(0, exp))
        return

    heur = {"v": 100 * f[metric]["mean"], "lo": 100 * f[metric]["lo"], "hi": 100 * f[metric]["hi"]}
    groups = []
    for meth in [m for m in METHOD_ORDER if any(r["method"] == m for _, r in runs)]:
        bars = []
        for model in [m for m in ("qwen7b", "qwen3b") if m in models]:
            hit = next(((k, r) for k, r in runs if r["method"] == meth and r["model"] == model), None)
            if not hit:
                continue
            k, r = hit
            s, dv = r[metric], e1[k]["vs_frequency"][metric]
            part = not r["complete"]
            bars.append({
                "v": 100 * s["mean"], "lo": 100 * s["lo"], "hi": 100 * s["hi"], "fill": ui.MODEL_VAR[model],
                "hatch": part, "hatch_cls": ui.MODEL_HATCH[model], "sub": ui.MODEL_SHORT[model],
                "subnote": f"partial {r['n']}/{exp}" if part else "",
                "tip": [f'{ui.sw(ui.MODEL_SW[model], False)}{ui.METHOD_SHORT[meth]} · {ui.MODELS[model]}',
                        f'{METRICS[metric]} {ui.pct(s)}', f'<span class="tx-ci">n = {r["n"]} · Δ {ui.num(100 * dv["diff"], signed=True)} vs heuristic</span>']})
        if bars:
            groups.append({"label": ui.METHODS[meth], "bars": bars})

    items = [(ui.sw(ui.MODEL_SW[m]), ui.MODELS[m]) for m in ("qwen7b", "qwen3b") if m in models
             and any(r["model"] == m for _, r in runs)]
    items.append(('<span class="lg-band"></span>', f"Frequency heuristic {ui.pct(f[metric])}"))
    if any(not r["complete"] for _, r in runs):
        items.append(('<span class="lg-hatch"></span>', "Partial run"))
    items.append(('<span class="lg-ci"></span>', "95% CI"))
    st.html(ui.legend(items))
    st.html(ui.bar_chart(groups, heur, height=300, aria=f"{METRIC_LONG[metric]} on {ui.DATASETS[dataset]}"))
    ui.caption("How to read this: a bar whose CI whisker overlaps the grey band is not distinguishable from simply "
               "counting the user’s past visits.")

    ui.section(f"Results · {ui.DATASETS[dataset]}",
               "All metrics in %, 95% CI in brackets · Δ paired on the same instances · p = McNemar, Holm-adjusted")
    rows = [{"cls": "ref", "cells": [
        "Frequency heuristic", f'<span style="display:inline-flex;align-items:center;gap:6px">{ui.sw("sw-h", False)}no LLM</span>',
        str(f["n"]), ui.pct(f["acc1"]), ui.pct(f["acc5"]), ui.pct(f["mrr"]), ui.pct(f["ndcg5"]),
        ui.num(100 * f["copy"]["mean"]), "0.0", ("–", "strong"), "–"]}]
    for k, r in sorted(runs, key=lambda kr: (METHOD_ORDER.index(kr[1]["method"]), kr[1]["model"] != "qwen7b")):
        v = e1[k]["vs_frequency"]
        name = ui.METHODS[r["method"]] + (f' {ui.partial_tag()}' if not r["complete"] else "")
        rows.append({"cells": [
            f'<span style="display:inline-flex;align-items:center;gap:8px">{name}</span>',
            f'<span style="display:inline-flex;align-items:center;gap:6px">{ui.sw(ui.MODEL_SW[r["model"]], False)}{ui.MODELS[r["model"]]}</span>',
            str(r["n"]) if r["complete"] else f'{r["n"]}/{exp}', ui.pct(r["acc1"]), ui.pct(r["acc5"]), ui.pct(r["mrr"]),
            ui.pct(r["ndcg5"]), ui.num(100 * r["copy"]["mean"]), ui.num(100 * r["invalid"]["mean"]),
            (ui.diff(v["acc1"]), "strong"), ui.pval(v["mcnemar_acc1"].get("p_holm"))]})
    st.html(ui.table([("Method", "l"), ("Model", "l"), ("n", "r"), ("Acc@1", "r"), ("Acc@5", "r"), ("MRR", "r"),
                      ("NDCG@5", "r"), ("Copy %", "r"), ("Invalid %", "r"), ("Δ Acc@1 vs heuristic", "r"),
                      ("p (Holm)", "r")], rows, min_width=1100))
    notes = ["Copy % = top prediction is a place from the user’s history. Both models run in 4-bit on the same GPU."]
    for k, r in runs:
        if not r["complete"]:
            notes.append(f"{ui.METHOD_SHORT[r['method']]} {ui.MODEL_SHORT[r['model']]} is compared with the heuristic "
                         f"on its {r['n']} completed instances only.")
    if dataset in C.NAME_DATASETS:
        notes.append("Massive-STEPS rows use the names-present condition; see E3 for the other conditions.")
    ui.caption(" ".join(notes))


def _engine_block(dataset: str):
    """Stronger non-LLM baselines and the assistant's evidence engine (Member 2's reports/assistant/results.json)."""
    ui.section(f"Stronger baselines and the assistant engine · {ui.DATASETS[dataset]}",
               "non-LLM rankers on the same instances · Δ vs the time-aware heuristic, p = McNemar, Holm-adjusted")
    rep = data.assistant_report("results")
    d = (rep or {}).get("datasets", {}).get(dataset)
    if not d:
        ui.caption("Appears when the offline evaluation of the assistant engine is in the repo "
                   "(reports/assistant/results.json, written by <code>python -m tracex_assist.evaluate</code>).")
        return
    rows = []
    for key, m in d.get("methods", {}).items():
        v = m.get("vs_frequency")
        name = ui.esc(m.get("label", key)) + (f" {ui.tag('post hoc')}" if m.get("exploratory") else "")
        rows.append({"cls": "ref" if key == "frequency" else "", "cells": [
            name, str(m.get("n", d.get("n", "–"))), ui.pct(m.get("acc1")), ui.pct(m.get("acc5")), ui.pct(m.get("mrr")),
            ui.pct(m.get("ndcg5")), (ui.diff(v["acc1"]) if v else "–", "strong"), ui.pval(v.get("p_holm")) if v else "–"]})
    st.html(ui.table([("Ranker", "l"), ("n", "r"), ("Acc@1", "r"), ("Acc@5", "r"), ("MRR", "r"), ("NDCG@5", "r"),
                      ("Δ Acc@1 vs heuristic", "r"), ("p (Holm)", "r")], rows, min_width=900))
    cat = d.get("category")
    if cat:
        crow = [{"cells": [ui.esc(m.get("label", k)), ui.pct(m.get("acc1")), ui.pct(m.get("acc5"))]} for k, m in cat.items()]
        st.html('<div class="tx-root tx-h3" style="margin-top:8px">Kind of place (category) prediction</div>' +
                ui.table([("Ranker", "l"), ("Category Acc@1", "r"), ("Category Acc@5", "r")], crow))
    ui.caption("Markov = where the user went next after their current place; recency = most recently visited places; "
               "engine = the assistant's fused ranking. Rows tagged post hoc were designed after seeing the LLM results.")


def render():
    ui.page_header("E1 Heuristic floor", "Does the LLM beat a counter that ranks each user’s most-visited places?")
    c1, c2, c3, c4 = st.columns([2.2, 1.6, 1, 1], vertical_alignment="bottom")
    with c1:
        ds = ui.segmented("Dataset", {**ui.DATASET_SHORT, "all": "All"}, "e1_ds")
    with c2:
        metric = ui.segmented("Metric", METRICS, "e1_metric")
    with c3:
        model = ui.select("Model", {"both": "Both models", **ui.MODELS}, "e1_model")
    with c4:
        meth = ui.select("Method", {"all": "All methods", "llmmob": "LLM-Mob", "retrieval-jaccard": "Retrieval (Jaccard)"},
                         "e1_method")
    ui.rule()
    sample, sig, res = context.get()
    models = ["qwen7b", "qwen3b"] if model == "both" else [model]
    methods = METHOD_ORDER if meth == "all" else [meth]
    for d in (list(ui.DATASETS) if ds == "all" else [ds]):
        _dataset_block(d, metric, models, methods, res, sample)
        _engine_block(d)
