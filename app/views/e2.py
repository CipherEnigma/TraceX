"""Page 3: E2 revisit vs new targets."""
from __future__ import annotations

import streamlit as st

import context
import data
import ui
from views.e1 import METHOD_ORDER


def render():
    ui.page_header("E2 Revisit vs new", "Can the LLM predict a place the user has never been, or only places it has seen?")
    c1, c2, c3, _ = st.columns([2, 1.4, 1, 1], vertical_alignment="bottom")
    with c1:
        ds = ui.segmented("Dataset", ui.DATASET_SHORT, "e2_ds")
    with c2:
        model = ui.segmented("Model", ui.MODELS, "e2_model")
    with c3:
        meth = ui.select("Method", {"all": "All methods", "llmmob": "LLM-Mob", "retrieval-jaccard": "Retrieval (Jaccard)"},
                         "e2_method")
    ui.rule()
    sample, sig, res = context.get()
    f = data.freq(res, ds, sample)
    nrev, nnew = f["revisit"]["n"], f["new"]["n"]
    tot = nrev + nnew
    share = 100 * nrev / tot if tot else 0.0

    st.html(
        '<div class="tx-grid tx-root" style="grid-template-columns:repeat(auto-fit,minmax(min(280px,100%),1fr))">'
        f'<div class="tx-tile" style="gap:4px"><span class="lbl">{ui.sw("sw-7", False)}Revisit targets</span>'
        f'<div style="display:flex;align-items:baseline;gap:8px"><span class="tx-big">{ui.num(share)}%</span>'
        f'<span class="tx-ci">n = {nrev} of {tot}</span></div>'
        f'<span class="tx-note">True next place appears in the user’s history</span></div>'
        f'<div class="tx-tile" style="gap:4px"><span class="lbl">{ui.sw("sw-3", False)}New targets</span>'
        f'<div style="display:flex;align-items:baseline;gap:8px"><span class="tx-big">{ui.num(100 - share)}%</span>'
        f'<span class="tx-ci">n = {nnew} of {tot}</span></div>'
        f'<span class="tx-note">True next place never visited before</span></div>'
        f'<div class="tx-tile" style="justify-content:center;gap:8px"><span class="tx-note">Split of {tot} '
        f'{ui.DATASET_SHORT[ds]} test targets</span><div role="img" aria-label="{ui.num(share)}% revisit" '
        f'style="display:flex;height:8px;border-radius:2px;overflow:hidden;gap:2px"><span style="width:{share}%;'
        f'background:var(--s7)"></span><span style="flex:1 1 auto;background:var(--s3)"></span></div>'
        f'<div class="tx-row tx-note"><span>{nrev} revisit</span><span>{nnew} new</span></div></div></div>')

    entries = [x for x in res["e2"] if x["dataset"] == ds and x["model"] == model
               and (meth == "all" or x["method"] == meth)]
    runs = {x["run"]: res["runs"][x["run"]] for x in entries}
    ui.section(f"Accuracy@1 by target type · {ui.DATASET_SHORT[ds]} · {ui.MODELS[model]}",
               ui.legend([(ui.sw("sw-7"), f"Revisit (n = {nrev})"), (ui.sw("sw-3"), f"New (n = {nnew})"),
                          ('<span class="lg-ci"></span>', "95% CI")]))

    def bars(summary, partial=False):
        out = []
        for tag, fill, hatch in (("revisit", "var(--s7)", "hatch-7"), ("new", "var(--s3)", "hatch-3")):
            s = summary[tag]["acc1"]
            b = {"v": 100 * s["mean"], "lo": 100 * s["lo"], "hi": 100 * s["hi"], "fill": fill, "hatch": partial,
                 "hatch_cls": hatch, "sub": "Revisit" if tag == "revisit" else "New",
                 "tip": [f'{"Revisit" if tag == "revisit" else "New"} targets', f"Acc@1 {ui.pct(s)}",
                         f'<span class="tx-ci">n = {summary[tag]["n"]}</span>']}
            if tag == "new":  # the finding: label value and hit count
                b["big"] = (100 * s["mean"], f'{round(s["mean"] * summary[tag]["n"])}/{summary[tag]["n"]}')
            out.append(b)
        return out

    groups = [{"label": "Frequency heuristic", "bars": bars(f)}]
    for m in METHOD_ORDER:
        hit = next(((k, r) for k, r in runs.items() if r["method"] == m), None)
        if hit:
            groups.append({"label": ui.METHODS[m], "bars": bars(hit[1], not hit[1]["complete"])})
    st.html(ui.bar_chart(groups, None, height=280, label_mode="value",
                         aria=f"Revisit vs new Accuracy@1 on {ui.DATASETS[ds]}"))
    if not runs:
        ui.caption(f"No {ui.MODELS[model]} results for {ui.DATASETS[ds]} yet: only the heuristic is shown.")
    ui.caption("New-target values are labelled because they sit near zero. The frequency heuristic can only rank "
               "visited places, so it scores 0 on new targets by construction.")

    ui.section("LLM vs heuristic within each subset", "Δ paired on the same instances · p = McNemar, Holm-adjusted")
    rows = []
    for x in sorted(entries, key=lambda x: (METHOD_ORDER.index(x["method"]), x["subset"] != "revisit")):
        r, v, tag = runs[x["run"]], x["vs_frequency"], x["subset"]
        rows.append({"cells": [
            ui.METHODS[x["method"]] + (f" {ui.partial_tag()}" if not r["complete"] else ""), ui.MODELS[model],
            f'<span style="display:inline-flex;align-items:center;gap:6px">{ui.sw("sw-7" if tag == "revisit" else "sw-3", False)}'
            f'{tag.title()}</span>', str(v["n"]), ui.pct(r[tag]["acc1"]), ui.pct(f[tag]["acc1"]),
            (ui.diff(v["acc1"]), "strong"), ui.pval(v["mcnemar_acc1"].get("p_holm"))]})
    if rows:
        st.html(ui.table([("Method", "l"), ("Model", "l"), ("Subset", "l"), ("n", "r"), ("LLM Acc@1", "r"),
                          ("Heuristic Acc@1", "r"), ("Δ", "r"), ("p", "r")], rows, min_width=900))
    else:
        ui.empty_state("No paired comparisons yet", "They appear once an LLM run for this dataset and model is in "
                       "results/llm/.", f"python -m tracex_llm.run --dataset {ds} --model {model}")

    with st.expander("Definition of history", expanded=True):
        st.html('<div class="tx-root" style="display:flex;flex-direction:column;gap:8px;color:var(--ink2);max-width:760px">'
                "<p style='margin:0'>History = the user’s training trajectories that end before the target trajectory "
                "begins, plus the check-ins of the current trajectory up to the prediction point.</p>"
                "<p style='margin:0'>A target is a <strong style='color:var(--ink);font-weight:500'>revisit</strong> if "
                "its POI id appears anywhere in that history, and <strong style='color:var(--ink);font-weight:500'>new"
                "</strong> otherwise. The same definition decides the “seen” tag in the Instance explorer and the "
                "copy rate.</p></div>")
