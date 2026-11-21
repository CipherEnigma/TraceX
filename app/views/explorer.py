"""Page 5: Instance explorer - one prediction end to end."""
from __future__ import annotations

import hashlib
from datetime import datetime

import streamlit as st

import context
import data
import ui
from tracex_llm import config as C
from tracex_llm.parse import user_poi_ids
from tracex_llm.prompts import build_prompt, present_names
from views.e1 import METHOD_ORDER
from views.e3 import aliases

WD = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
WD_LONG = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
FILTERS = ["All", "LLM hit", "LLM miss", "Revisit", "New", "Methods disagree"]


def name_fn(dataset: str, cond: str):
    if dataset not in C.NAME_DATASETS or cond in ("removed", C.NO_NAMES):
        return None
    return present_names if cond == "present" else aliases().name_fn


@st.cache_data(show_spinner=False)
def freq_rankings(dataset: str, sample: str) -> dict:
    ds = data.dataset(dataset)["ds"]
    return {i["instance_id"]: data.frequency_ranking(dataset, i) for i in ds.instances(sample)}


def _row(c: dict, nf, hl=False) -> str:
    name = nf(c) if nf else ""
    cat = c["category"]
    return (f'<div class="tx-ci-row{" hl" if hl else ""}"><span class="t">{c["time"][11:16]}</span>'
            f'<span class="p">{c["poi_id"]}</span><span title="{ui.esc(cat)}">{ui.esc(cat)}</span>'
            f'<span class="n" title="{ui.esc(name or "")}">{ui.esc(name or "")}</span></div>')


def _day(t: str) -> str:
    d = datetime.strptime(t[:10], "%Y-%m-%d")
    return f"{t[:10]} {WD_LONG[d.weekday()]}"


def _pred_card(title: str, swatch: str, ranked: list | None, target: int, seen: set, pois: dict,
               reveal: bool, pending: str = "") -> str:
    head = f'<div class="nm">{swatch}{ui.esc(title)}</div>'
    if ranked is None:
        return (f'<div class="tx-pred"><div class="tx-predhead">{head}<div class="tags">{ui.partial_tag()}</div></div>'
                f'<div class="tx-pending"><span>Not predicted yet.</span><span class="tx-note">{pending}</span></div></div>')
    rank = ranked.index(target) + 1 if target in ranked else 0
    if reveal:
        h1 = ui.tag("✓ hit@1", "ok") if rank == 1 else ui.tag("✗ hit@1", "bad")
        h5 = ui.tag("✓ hit@5", "ok") if 1 <= rank <= 5 else ui.tag("✗ hit@5", "bad")
    else:
        h1, h5 = ui.tag("hit@1 hidden"), ui.tag("hit@5 hidden")
    if not ranked:  # unparseable / empty answer: an invalid prediction, shown as a block, not a list row
        return (f'<div class="tx-pred"><div class="tx-predhead">{head}<div class="tags">{h1}{h5}'
                f'{ui.tag("invalid", "bad")}</div></div><div class="tx-pending"><span>No usable prediction.</span>'
                f'<span class="tx-note">The answer could not be parsed into POI ids; see the raw output below. '
                f'It counts as a miss and in the invalid rate.</span></div></div>')
    copy = ui.tag("copy" if ranked[0] in seen else "no copy")
    items = []
    for i, p in enumerate(ranked, 1):
        valid = p in pois
        hit = reveal and p == target
        tags = (('<span class="mini ok">✓</span>' if hit else "") + ('<span class="mini">seen</span>' if p in seen else "")
                + ("" if valid else '<span class="mini bad">invalid</span>'))
        cat = pois[p][0] if valid else "not in POI set"
        items.append(f'<li class="{"hit" if hit else ""}"><span class="rk">{i}</span>'
                     f'<span class="id {"" if valid else "inv"}">{p}</span><span class="cat" title="{ui.esc(cat)}">'
                     f'{ui.esc(cat)}</span><span class="tg">{tags}</span></li>')
    return (f'<div class="tx-pred"><div class="tx-predhead">{head}<div class="tags">{h1}{h5}{copy}</div></div>'
            f'<ol>{"".join(items)}</ol></div>')


def render():
    ui.page_header("Instance explorer", "Inspect one prediction end to end: the history the model saw, what each "
                                        "method ranked, and why.")
    c1, c2, c3, _ = st.columns([2.2, 1, 1, 1], vertical_alignment="bottom")
    with c1:
        ds_key = ui.segmented("Dataset", ui.DATASET_SHORT, "ex_ds")
    with c2:
        shown = ui.select("Models shown", {"both": "7B + 3B", "qwen7b": "7B", "qwen3b": "3B"}, "ex_models")
    with c3:
        has_names = ds_key in C.NAME_DATASETS
        cond = ui.select("Names", {c: c for c in C.NAME_CONDITIONS} if has_names else {C.NO_NAMES: "none in this dataset"},
                         f"ex_cond_{has_names}", disabled=not has_names)
    ui.rule()
    sample, sig, res = context.get()
    d = data.dataset(ds_key)
    ds, pois = d["ds"], d["pois"]
    insts = ds.instances(sample)
    recs = data.run_records(sample, sig)
    models = ["qwen7b", "qwen3b"] if shown == "both" else [shown]
    llm_runs = []  # (key, method, model)
    for m in METHOD_ORDER[:2]:
        for model in models:
            k = data.run_key({"dataset": ds_key, "method": m, "model": model, "condition": cond}, sample)
            if k in recs:
                llm_runs.append((k, m, model))
    franks = freq_rankings(ds_key, sample)

    def top1s(i):
        out = [franks[i["instance_id"]][0]] if franks[i["instance_id"]] else []
        for k, _, _ in llm_runs:
            r = recs[k].get(i["instance_id"])
            if r and r["parse"]["ranked"]:
                out.append(r["parse"]["ranked"][0])
        return out

    def llm_hit(i):
        rs = [recs[k].get(i["instance_id"]) for k, _, _ in llm_runs]
        rs = [r for r in rs if r]
        return None if not rs else any(r["parse"]["ranked"][:1] == [r["target"]] for r in rs)

    tests = {"All": lambda i: True, "LLM hit": lambda i: llm_hit(i) is True, "LLM miss": lambda i: llm_hit(i) is False,
             "Revisit": lambda i: i["is_revisit"], "New": lambda i: not i["is_revisit"],
             "Methods disagree": lambda i: len(set(top1s(i))) > 1}
    counts = {f: sum(1 for i in insts if tests[f](i)) for f in FILTERS}

    left, right = st.columns([1, 3.3], gap="large")
    with left:
        q = st.text_input("Instance", key="ex_q", placeholder="Search by instance or user ID",
                          help="e.g. nyc-test-22, or u_1 for user 1")
        flt = st.radio("Quick filter", FILTERS, key="ex_filter", format_func=lambda f: f"{f} · {counts[f]}")
        pool = [i for i in insts if tests[flt](i)]
        if q.strip():
            qq = q.strip().lower()
            pool = [i for i in pool if qq in i["instance_id"].lower() or qq == f"u_{i['user_id']}" or qq == str(i["user_id"])]
        sel_key = f"ex_pos_{ds_key}_{flt}_{q}"
        pos = st.session_state.get(sel_key, 0)
        b1, b2 = st.columns(2)
        if b1.button("Previous", icon=":material/chevron_left:", width="stretch", disabled=not pool):
            pos = (pos - 1) % len(pool)
        if b2.button("Next", icon=":material/chevron_right:", width="stretch", disabled=not pool):
            pos = (pos + 1) % len(pool)
        pos = min(pos, max(len(pool) - 1, 0))
        st.session_state[sel_key] = pos
        if pool:
            st.html(f'<div class="tx-note" style="text-align:center">{pos + 1} of {len(pool)} in “{ui.esc(flt)}”</div>')

    with right:
        if not pool:
            what = " · ".join(f"{ui.METHOD_SHORT[m]} {ui.MODEL_SHORT[mo]}" for _, m, mo in llm_runs) or "no LLM runs"
            ui.empty_state(f"No instance matches “{flt}” for {ui.DATASETS[ds_key]}",
                           f"Shown runs: {ui.esc(what)}. If a filter like LLM hit is empty, the runs have not ranked the "
                           "true place first on any finished instance yet. That is a finding, not an error. Clear the "
                           "search or pick another quick filter.",
                           status_html=f'<span class="tx-status st-run">○ 0 of {len(insts)} instances</span>')
            return
        inst = pool[pos]
        hist = ds.history(inst)
        seen = user_poi_ids(inst, hist)
        target = inst["target"]["poi_id"]
        tt = inst["target"]["time"]
        nf = name_fn(ds_key, cond)
        n_ci = sum(len(t["checkins"]) for t in hist)
        kind = ("sw-7", "Revisit") if inst["is_revisit"] else ("sw-3", "New")
        st.html(
            f'<div class="tx-head tx-root">'
            f'<div><span class="k">Instance</span><span class="tx-mono" style="font-size:16px;font-weight:500">{inst["instance_id"]}</span></div>'
            f'<div><span class="k">User</span><span class="tx-mono">u_{inst["user_id"]}</span></div>'
            f'<div><span class="k">Target time</span><span class="tx-mono">{tt[:16]} {WD[inst["target"]["weekday"]]}</span></div>'
            f'<div><span class="k">Target type</span>{ui.tag(ui.sw(kind[0], False) + kind[1])}</div>'
            f'<div><span class="k">History</span>{len(hist)} trajectories <span class="tx-ci">· {n_ci} check-ins</span></div></div>')

        _, tcol = st.columns([3, 1])
        reveal = tcol.toggle("Reveal target", key="ex_reveal", value=False)
        hl = lambda c: reveal and c["poi_id"] == target  # noqa: E731
        hist_html = []
        for t in reversed(hist):
            hist_html.append(f'<div class="tx-day"><span>{_day(t["checkins"][0]["time"])}</span>'
                             f'<span class="tx-mono" style="font-size:12px;font-weight:400">traj {t["traj_id"]}</span></div>')
            hist_html += [_row(c, nf, hl(c)) for c in t["checkins"]]
        prefix = inst["prefix"]
        if reveal:
            tc = inst["target"]
            tname = nf(tc) if nf else ""
            trow = (f'<div class="tx-target"><span class="tx-mono" style="color:var(--ink2)">{tt[11:16]}</span>'
                    f'<span class="tx-mono" style="text-align:right;font-weight:600">{target}</span>'
                    f'<span style="font-weight:500">{ui.esc(tc["category"])}</span><span class="tx-ci">{ui.esc(tname or "")}</span>'
                    f'<span style="font-size:12px;color:var(--ok);font-weight:500">✓ Target</span></div>')
            k = sum(c["poi_id"] == target for t in hist for c in t["checkins"])
            foot = (f"The target POI {target} appears {k} time{'s' if k != 1 else ''} in this user’s history."
                    if k else f"The target POI {target} never appears in this user’s history: a new place.")
        else:
            trow = (f'<div class="tx-target hidden"><span class="tx-mono">{tt[11:16]}</span>'
                    f'<span class="tx-mono" style="text-align:right">?</span><span>Next: ?</span></div>')
            foot = "Turn on Reveal target to score the predictions."
        st.html(
            '<div class="tx-root" style="display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(min(340px,100%),1fr))">'
            f'<section class="tx-panel"><div class="tx-panelhead"><h2 class="tx-h3">History</h2>'
            f'<span class="tx-note">newest first · {len(hist)} trajectories</span></div>'
            f'<div class="tx-scroll">{"".join(hist_html)}</div></section>'
            f'<section class="tx-panel"><div class="tx-panelhead"><h2 class="tx-h3">Current trajectory</h2>'
            f'<span class="tx-note">{len(prefix)} check-in{"s" if len(prefix) != 1 else ""} so far</span></div>'
            f'<div style="padding:8px 0"><div class="tx-day" style="padding-top:0">{_day(prefix[0]["time"])}</div>'
            f'{"".join(_row(c, nf, hl(c)) for c in prefix)}{trow}</div>'
            f'<div class="tx-panelfoot">{foot}</div></section></div>')

        ui.section("Predictions · top 10", '<span style="display:flex;gap:12px"><span>seen = in user history</span>'
                                           '<span>copy = top-1 is a seen place</span></span>')
        cards = [_pred_card("Frequency", ui.sw("sw-h", False), franks[inst["instance_id"]], target, seen, pois, reveal)]
        for k, m, model in llm_runs:
            r = recs[k].get(inst["instance_id"])
            title = f"{ui.METHOD_SHORT[m]} · {ui.MODEL_SHORT[model]}"
            if r is None:
                order = [i["instance_id"] for i in insts].index(inst["instance_id"]) + 1
                cards.append(_pred_card(title, ui.sw(ui.MODEL_SW[model], False), None, target, seen, pois, reveal,
                                        f"{len(recs[k])}/{len(insts)} done; this instance is #{order} in the run order."))
            else:
                cards.append(_pred_card(title, ui.sw(ui.MODEL_SW[model], False), r["parse"]["ranked"], target, seen,
                                        pois, reveal))
        st.html(f'<div class="tx-preds tx-root">{"".join(cards)}</div>')
        if not llm_runs:
            ui.caption(f"No LLM results for {ui.DATASETS[ds_key]}"
                       f"{' with names ' + cond if has_names else ''} in this sample yet; only the heuristic is shown.")
            return

        ui.section("LLM answer")
        answered = [(k, m, model) for k, m, model in llm_runs if inst["instance_id"] in recs[k]]
        if not answered:
            ui.caption("None of the shown runs has reached this instance yet.")
            return
        tabs = st.tabs([f"{ui.METHOD_SHORT[m]} {ui.MODEL_SHORT[model]}" for _, m, model in answered])
        for tab, (k, m, model) in zip(tabs, answered):
            r = recs[k][inst["instance_id"]]
            with tab:
                reason = r["parse"].get("reason") or "(no reason given)"
                st.html(f'<figure class="tx-quote tx-root"><blockquote>“{ui.esc(reason)}”</blockquote>'
                        f'<figcaption><span>{ui.METHODS[m]} · {ui.MODELS[model]}</span>'
                        f'<span>parse: {ui.esc(r["parse"]["status"])}</span>'
                        f'<span>{r.get("new_tokens") or "–"} output tokens</span>'
                        f'<span>{ui.num(r.get("seconds"), 2)} s</span></figcaption></figure>')
                with st.expander("Raw output", expanded=True):
                    st.html(f'<pre class="tx-code">{ui.esc(r["raw"])}</pre>')
                method, retr = data.method_parts(m)
                p = build_prompt(method, inst, hist, d["idf"], retr or C.DEFAULT_RETRIEVER, nf)
                same = hashlib.sha1(p.user.encode()).hexdigest()[:12] == r.get("prompt_sha1")
                check = ("✓ identical to the prompt used in the run" if same else
                         "✗ differs from the run's prompt (settings changed since the run)")
                with st.expander(f"Full prompt · {r.get('prompt_tokens') or '–'} tokens · {check}", expanded=False):
                    st.code(f"SYSTEM\n{p.system}\n\nUSER\n{p.user}", language=None, wrap_lines=True, height=240)
