"""Page 1: Overview - the whole story on one screen."""
from __future__ import annotations

import streamlit as st

import context
import data
import ui
from tracex_llm import config as C

SOURCES = [
    ("Copying", "The model repeats places the user already visits, so it should track the frequency heuristic "
                "and fail on places never visited.", [("E1", "e1"), ("E2", "e2")]),
    ("Recall", "The model uses memorised knowledge of real venues, so accuracy should fall when names are "
               "removed or swapped for aliases.", [("E3", "e3")]),
    ("Reasoning", "The model infers the next place from time, order and category, so it should beat the "
                  "heuristic and get some new places right.", [("E1", "e1"), ("E2", "e2")]),
]


def _verdict(d: dict, dataset: str) -> str:
    if d["lo"] > 0:
        return "Above the floor" + ("; see E3 for how much of it needs venue names." if dataset in C.NAME_DATASETS
                                    else ": the CI excludes zero.")
    if d["hi"] < 0:
        return "Below the floor: the counter wins."
    return "CI crosses zero: no better than counting visits."


def _tile(dataset: str, res: dict, sample: str) -> str:
    f = data.freq(res, dataset, sample)
    head = (f'<div class="tx-row" style="align-items:baseline"><span style="font-weight:600">{ui.DATASETS[dataset]}</span>'
            f'<span class="tx-note">n = {f["n"]}</span></div>')
    heur = (f'<div class="tx-row"><span class="lbl" style="font-size:13px"><span style="width:12px;border-top:1.5px dashed var(--heur)">'
            f'</span>Frequency heuristic</span><span>{ui.pct(f["acc1"])}</span></div>')
    cands = [x for x in res["e1"] if x["dataset"] == dataset]
    if not cands:
        return (f'<div class="tx-tile">{head}<div style="display:flex;flex-direction:column;gap:4px;font-size:13px">{heur}'
                f'<div class="tx-row"><span class="tx-ci">No LLM run yet</span><span class="tx-ci">–</span></div></div>'
                f'<div class="tx-note" style="border-top:1px solid var(--border);padding-top:12px">'
                f'Run the e1 stage on Kaggle to compare.</div></div>')
    best = max(cands, key=lambda x: res["runs"][x["run"]]["acc1"]["mean"])
    r, v = res["runs"][best["run"]], best["vs_frequency"]
    d = v["acc1"]
    name = f'{ui.METHOD_SHORT[r["method"]]} · {ui.MODEL_SHORT[r["model"]]}'
    if r["condition"] == "present":
        name += " · names present"
    part = " " + ui.partial_tag() if not r["complete"] else ""
    clamp = lambda x: max(0.0, min(100.0, 50 + 5 * 100 * x))  # noqa: E731  (-10..+10 pp scale)
    lo, hi, pt = clamp(d["lo"]), clamp(d["hi"]), clamp(d["diff"])
    llm = (f'<div class="tx-row"><span class="lbl" style="font-size:13px">{ui.sw(ui.MODEL_SW[r["model"]], False)}'
           f'{ui.esc(name)}{part}</span><span>{ui.pct(r["acc1"])}</span></div>')
    return (
        f'<div class="tx-tile">{head}<div style="display:flex;flex-direction:column;gap:4px;font-size:13px">{heur}{llm}</div>'
        f'<div style="display:flex;align-items:baseline;gap:8px;border-top:1px solid var(--border);padding-top:12px">'
        f'<span class="tx-big">{ui.num(100 * d["diff"], signed=True)}</span>'
        f'<span class="tx-ci">[{ui.num(100 * d["lo"], signed=True)}, {ui.num(100 * d["hi"], signed=True)}]</span>'
        f'<span class="tx-note" style="margin-left:auto">p = {ui.pval(v["mcnemar_acc1"].get("p_holm"))}</span></div>'
        f'<div style="position:relative;height:16px" role="img" aria-label="Difference {ui.num(100 * d["diff"], signed=True)} '
        f'percentage points, CI {ui.num(100 * d["lo"], signed=True)} to {ui.num(100 * d["hi"], signed=True)}">'
        f'<div style="position:absolute;left:0;right:0;top:8px;border-top:1px solid var(--border)"></div>'
        f'<div style="position:absolute;left:50%;top:2px;height:12px;border-left:1px solid var(--ink2)"></div>'
        f'<div style="position:absolute;top:7px;height:3px;border-radius:2px;background:var(--ink);left:{lo}%;width:{max(hi - lo, 0.5)}%"></div>'
        f'<div style="position:absolute;top:4px;width:9px;height:9px;margin-left:-4px;border-radius:50%;background:var(--bg);'
        f'border:1.5px solid var(--ink);box-sizing:border-box;left:{pt}%"></div></div>'
        f'<div class="tx-row tx-note"><span>{ui.MINUS}10</span><span>0</span><span>+10 pp</span></div>'
        f'<div class="tx-note">{_verdict(d, dataset)}</div></div>')


def render():
    ui.page_header("Overview", "When Qwen2.5 predicts a person’s next check-in, is it copying their habits, "
                               "recalling memorised venues, or reasoning?")
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.6], vertical_alignment="bottom")
    with c1:
        ds_f = ui.select("Dataset", {"all": "All datasets", **ui.DATASETS}, "ov_ds")
    with c2:
        model_f = ui.select("Model", {"all": "All models", **ui.MODELS}, "ov_model")
    with c3:
        meth_f = ui.select("Method", {"all": "All methods", "llmmob": "LLM-Mob", "retrieval-jaccard": "Retrieval (Jaccard)"},
                           "ov_method")
    sample, sig, res = context.get()
    datasets = list(ui.DATASETS) if ds_f == "all" else [ds_f]
    total = sum(data.expected_n(d, sample) for d in ui.DATASETS)
    with c4:
        st.html(f'<div class="tx-note" style="text-align:right;padding-bottom:10px">{total:,} test instances · '
                f'3 datasets · 2 models</div>')
    ui.rule()

    st.html('<div class="tx-grid tx-root" style="gap:32px">' + "".join(
        f'<div class="tx-source"><div class="tx-h2">{t}</div><p>{p}</p><div class="tx-note" style="display:flex;gap:8px">'
        f'Tested by ' + "".join(f'<a href="/{u}" target="_self" style="color:var(--accent)">{lab}</a>' for lab, u in links)
        + "</div></div>" for t, p, links in SOURCES) + "</div>")

    ui.section("Best LLM vs frequency heuristic · Accuracy@1", "Δ is paired, in percentage points, 95% CI · McNemar p (Holm)")
    st.html('<div class="tx-grid tx-root">' + "".join(_tile(d, res, sample) for d in datasets) + "</div>")

    # Gate 1
    key = C.run_name("nyc", "qwen7b", "llmmob", sample=sample)
    if sample != "sample_test_500":
        verdict, color, detail = "Needs sample_test_500", "var(--ink2)", "switch the sample in the sidebar"
    elif key not in res["runs"]:
        verdict, color, detail = "Not run yet", "var(--ink2)", "run STAGE = \"gate1\" on Kaggle"
    else:
        r = res["runs"][key]
        a = 100 * r["acc1"]["mean"]
        detail = f'LLM-Mob 7B · {ui.pct(r["acc1"])} · n = {r["n"]}'
        if 18 <= a <= 23:
            verdict, color = "✓ Passed · within published range", "var(--ok)"
        elif a > 23:
            verdict, color = "✓ Passed · above published range", "var(--ok)"
        elif a >= 15:
            verdict, color = "◐ Within 3 points of the published range", "var(--ink2)"
        else:
            verdict, color = "✗ More than 3 points below: check the pipeline", "var(--bad)"
    st.html(f'<div class="tx-callout tx-root">{ui.tag("Gate 1")}<span>Reproduction check: NYC 7B Accuracy@1 vs published '
            f'18–23%</span><span class="tx-mono">{detail}</span><span style="margin-left:auto;color:{color};font-weight:500">'
            f'{verdict}</span></div>')

    # Experiment status
    rows, counts = [], {"done": 0, "part": 0, "none": 0}
    for p in data.plan():
        if p["dataset"] not in datasets or (model_f != "all" and p["model"] != model_f) or \
                (meth_f != "all" and p["method"] != meth_f):
            continue
        k = data.run_key(p, sample)
        exp = data.expected_n(p["dataset"], sample)
        done = res["runs"][k]["n"] if k in res["runs"] else 0
        counts["done" if done >= exp else "part" if done else "none"] += 1
        main = p["condition"] in (C.NO_NAMES, "present")
        names = "–" if p["condition"] == C.NO_NAMES else p["condition"]
        rows.append({"cells": [ui.DATASET_SHORT[p["dataset"]], ui.METHOD_SHORT[p["method"]], ui.MODELS[p["model"]],
                               (names, "muted"), ui.status(done, exp, main), ui.status(done, exp, main),
                               ui.status(done, exp, p["dataset"] in C.NAME_DATASETS)]})
    n = sum(counts.values())
    ui.section("Experiment status", f'<span style="display:flex;gap:16px"><span>{n} runs</span><span>✓ {counts["done"]} complete'
                                    f'</span><span>◐ {counts["part"]} partial</span><span>○ {counts["none"]} not run</span></span>')
    st.html(ui.table([("Dataset", "l"), ("Method", "l"), ("Model", "l"), ("Names", "l"), ("E1 Heuristic floor", "l"),
                      ("E2 Revisit vs new", "l"), ("E3 Name test", "l")], rows, min_width=880))
    ui.caption("The frequency heuristic is computed for every dataset and needs no run. E3 applies to Massive-STEPS "
               "Tokyo only; other cells show “n/a”. Massive-STEPS rows with names removed or aliased feed E3 only.")
