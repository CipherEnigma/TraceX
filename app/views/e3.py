"""Page 4: E3 venue-name test on Massive-STEPS Tokyo."""
from __future__ import annotations

import streamlit as st

import context
import data
import ui
from tracex_llm import config as C
from tracex_llm.aliasing import AliasMap
from tracex_llm.prompts import format_checkin
from views.e1 import METHOD_ORDER

DS = "msteps_tokyo"


@st.cache_resource(show_spinner=False)
def aliases():
    return AliasMap.for_dataset(DS)


def _diff_panels(inst: dict):
    am = aliases()
    lines = inst["prefix"][-4:]
    panels = [("present", "real venue names", lambda c: c.get("name")),
              ("removed", "name field dropped", None),
              ("aliased", "consistent fake aliases", am.name_fn)]
    out = []
    for cond, note, fn in panels:
        rows = []
        for c in lines:
            base = ui.esc(format_checkin(c, None))
            name = fn(c) if fn else None
            rows.append(f'<div>{base}{" | " if name else ""}'
                        f'{f"<span class=tx-namespan>&quot;{ui.esc(name)}&quot;</span>" if name else ""}</div>')
        out.append(f'<div class="tx-cond"><div class="tx-condhead">{ui.sw(ui.COND_SW[cond], False)}{ui.CONDITIONS[cond]}'
                   f'<span class="nt">{note}</span></div><div class="tx-condbody">{"".join(rows)}</div></div>')
    return ('<div class="tx-root" style="display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr))">'
            + "".join(out) + "</div>")


def render():
    ui.page_header("E3 Name test", "Massive-STEPS Tokyo: identical history and targets in every condition; "
                                   "only venue names change.")
    c1, c2, c3, _ = st.columns([1.3, 1.3, 1.6, 1], vertical_alignment="bottom")
    with c1:
        st.selectbox("Dataset", ["Massive-STEPS Tokyo · fixed"], disabled=True, key="e3_ds")
    with c2:
        model = ui.segmented("Model", ui.MODELS, "e3_model")
    with c3:
        meth = ui.segmented("Method", {"both": "Both", "llmmob": "LLM-Mob", "retrieval-jaccard": "Retrieval"}, "e3_method")
    ui.rule()
    sample, sig, res = context.get()
    exp = data.expected_n(DS, sample)
    f = data.freq(res, DS, sample)
    runs = [(k, r) for k, r in context.runs_for(res, DS, model) if meth == "both" or r["method"] == meth]

    ui.section(f"Accuracy@1 by name condition · {ui.MODELS[model]}",
               ui.legend([(ui.sw(ui.COND_SW[c]), ui.CONDITIONS[c]) for c in C.NAME_CONDITIONS]
                         + [('<span class="lg-band"></span>', f"Frequency heuristic {ui.pct(f['acc1'])}")]))
    if not runs:
        ui.empty_state(f"No E3 results yet for {ui.MODELS[model]}",
                       f"The name test runs each method three times on the same {exp} instances. The example below "
                       "already shows exactly what changes between conditions.",
                       f'# on Kaggle: STAGE = "e3", MODEL = "{model}"\n'
                       f"python -m tracex_llm.run --dataset msteps_tokyo --model {model} --condition all",
                       status_html=ui.status(0, exp))
    else:
        groups = []
        for m in METHOD_ORDER:
            bars = []
            for cond in C.NAME_CONDITIONS:
                hit = next((r for _, r in runs if r["method"] == m and r["condition"] == cond), None)
                if not hit:
                    continue
                s, part = hit["acc1"], not hit["complete"]
                bars.append({"v": 100 * s["mean"], "lo": 100 * s["lo"], "hi": 100 * s["hi"], "fill": ui.COND_VAR[cond],
                             "hatch": part, "hatch_cls": ui.COND_HATCH[cond], "sub": cond,
                             "subnote": f"partial {hit['n']}/{exp}" if part else "",
                             "tip": [f"{ui.sw(ui.COND_SW[cond], False)}{ui.METHOD_SHORT[m]} · {ui.CONDITIONS[cond]}",
                                     f"Acc@1 {ui.pct(s)}", f'<span class="tx-ci">n = {hit["n"]}</span>']})
            if bars:
                groups.append({"label": ui.METHODS[m], "bars": bars})
        heur = {"v": 100 * f["acc1"]["mean"], "lo": 100 * f["acc1"]["lo"], "hi": 100 * f["acc1"]["hi"]}
        st.html(ui.bar_chart(groups, heur, height=280, bar_w=60, gap=10, label_mode="value",
                             aria=f"E3 Accuracy@1 by name condition, {ui.MODELS[model]}"))
        ui.caption("If accuracy drops when names are removed and aliases do not recover it, the model relies on "
                   "knowledge of the real venues rather than on the history alone. This measures reliance on "
                   "names, not proof that the data was memorised.")

    comps = [x for x in res["e3"] if x["model"] == model and (meth == "both" or x["method"] == meth)]
    ui.section("Paired comparisons", "Δ in percentage points, 95% CI · p = McNemar (Acc@1 Holm-adjusted)")
    if comps:
        rows = []
        for x in sorted(comps, key=lambda x: METHOD_ORDER.index(x["method"])):
            part = x["n"] < exp
            rows.append({"cells": [
                ui.METHODS[x["method"]],
                f'<span style="display:inline-flex;align-items:center;gap:6px">{ui.sw(ui.COND_SW[x["a"]], False)}{x["a"]}'
                f'<span class="tx-ci">{ui.MINUS}</span>{ui.sw(ui.COND_SW[x["b"]], False)}{x["b"]}</span>',
                (f'{ui.partial_tag()} ' if part else "") + str(x["n"]), (ui.diff(x["acc1"]), "strong"),
                ui.pval(x["mcnemar_acc1"].get("p_holm")), (ui.diff(x["acc5"]), "strong"), ui.pval(x["mcnemar_acc5"]["p"])]})
        st.html(ui.table([("Method", "l"), ("Comparison", "l"), ("n paired", "r"), ("Δ Acc@1", "r"), ("p", "r"),
                          ("Δ Acc@5", "r"), ("p", "r")], rows, min_width=900))
    else:
        ui.caption("Comparisons appear once at least two name conditions have results for the same model and method.")

    # See the difference: a real instance under all three conditions
    d = data.dataset(DS)
    insts = d["ds"].instances(sample)
    multi = [i for i in insts if len(i["prefix"]) >= 3] or sorted(insts, key=lambda i: -len(i["prefix"]))[:20]
    ids = [i["instance_id"] for i in multi]
    ui.section("See the difference", "current trajectory excerpt · venue-name spans tinted")
    pick = st.selectbox("Example instance", ids, key="e3_example", label_visibility="collapsed")
    inst = next(i for i in multi if i["instance_id"] == pick)
    st.html(_diff_panels(inst))
