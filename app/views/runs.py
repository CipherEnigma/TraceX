"""Page 6: Runs & behaviour - progress, cost, settings, copy / invalid / parse rates."""
from __future__ import annotations

import streamlit as st

import context
import data
import ui
from tracex_llm import config as C

PLAN_INSTANCES = {"e1": 2 * 2 * 500, "e3": 2 * 3 * 334}  # per model, matches the Kaggle notebook


def render():
    ui.page_header("Runs & behaviour", "Every run’s progress, cost and settings, plus how often each model copies, "
                                       "fails or needs a fallback parser.")
    c1, c2, c3, _ = st.columns([1.2, 1, 1, 1.6], vertical_alignment="bottom")
    with c1:
        ds_f = ui.select("Dataset", {"all": "All datasets", **ui.DATASETS}, "ru_ds")
    with c2:
        model_f = ui.select("Model", {"all": "All models", **ui.MODELS}, "ru_model")
    with c3:
        meth_f = ui.select("Method", {"all": "All methods", "llmmob": "LLM-Mob", "retrieval-jaccard": "Retrieval (Jaccard)"},
                           "ru_method")
    ui.rule()
    sample, sig, res = context.get()
    meta = data.run_meta(sample, sig)

    plan = [p for p in data.plan() if (ds_f == "all" or p["dataset"] == ds_f)
            and (model_f == "all" or p["model"] == model_f) and (meth_f == "all" or p["method"] == meth_f)]
    rows, done_tot, exp_tot, states = [], 0, 0, {"part": 0, "none": 0}
    spi_by_model: dict[str, list] = {}
    for p in plan:
        k = data.run_key(p, sample)
        exp = data.expected_n(p["dataset"], sample)
        r = res["runs"].get(k)
        mt = meta.get(k, {})
        done = r["n"] if r else 0
        done_tot += min(done, exp)
        exp_tot += exp
        if done < exp:
            states["part" if done else "none"] += 1
        if mt.get("spi"):
            spi_by_model.setdefault(p["model"], []).append((done, mt["spi"]))
        icon = ("✓", "var(--ok)", "var(--ok)") if done >= exp else ("◐", "var(--ink2)", "var(--accent)") if done else \
            ("○", "var(--ink2)", "var(--border)")
        prog = (f'<span style="display:inline-flex;align-items:center;gap:8px;justify-content:flex-end">'
                f'<span style="font-size:12px;color:{icon[1]}">{icon[0]}</span><span style="min-width:64px">{done}/{exp}</span>'
                f'<span class="tx-prog"><span style="width:{100 * done / exp if exp else 0:.0f}%;background:{icon[2]}"></span></span></span>')
        rows.append({"cells": [
            ui.DATASET_SHORT[p["dataset"]],
            f'<span style="display:inline-flex;align-items:center;gap:6px">{ui.sw(ui.MODEL_SW[p["model"]], False)}{ui.MODELS[p["model"]]}</span>',
            ui.METHOD_SHORT[p["method"]], ("–" if p["condition"] == C.NO_NAMES else p["condition"], "muted"), prog,
            ui.num(mt.get("spi"), 2) if mt.get("spi") else "–",
            ui.num(r["mean_new_tokens"], 0) if r and r.get("mean_new_tokens") else "–",
            mt.get("gpu") or "–", (f'<span class="tx-mono">{mt.get("fingerprint") or "–"}</span>', "muted"),
            (data.fmt_when(mt.get("updated")), "muted")]})

    # Compute + progress tiles
    def wavg(xs):
        n = sum(a for a, _ in xs)
        return sum(a * b for a, b in xs) / n if n else None
    s7, s3 = wavg(spi_by_model.get("qwen7b", [])), wavg(spi_by_model.get("qwen3b", []))
    plan_n = sum(PLAN_INSTANCES.values())
    left_h = 0.0
    for p in plan:
        k = data.run_key(p, sample)
        rem = data.expected_n(p["dataset"], sample) - (res["runs"][k]["n"] if k in res["runs"] else 0)
        spi = (s7 if p["model"] == "qwen7b" else s3) or s7 or 3.74
        left_h += max(rem, 0) * spi / 3600
    gpu = next((m["gpu"] for m in meta.values() if m.get("gpu")), None)
    main = (f'<span class="tx-big">{ui.num(s7, 2)} s</span><span class="tx-ci">per instance · Qwen2.5-7B'
            f'{" on a " + ui.esc(gpu) if gpu else ""}</span>') if s7 else \
        '<span class="tx-big">–</span><span class="tx-ci">no timed 7B run yet</span>'
    extra = f'<span><span style="font-weight:500">{ui.num(s7 * plan_n / 3600, 1)} GPU-hours</span> <span class="tx-ci">' \
            f'projected per model, full plan</span></span>' if s7 else ""
    extra += f'<span><span style="font-weight:500">{ui.num(s3, 2)} s</span> <span class="tx-ci">3B</span></span>' if s3 else \
        '<span class="tx-ci">3B not timed yet</span>'
    pct_done = 100 * done_tot / exp_tot if exp_tot else 0
    st.html(
        '<div class="tx-grid tx-root" style="grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr))">'
        f'<div class="tx-tile" style="gap:8px"><span class="lbl">Compute</span><div style="display:flex;align-items:baseline;'
        f'gap:8px;flex-wrap:wrap">{main}</div><div style="display:flex;gap:24px;border-top:1px solid var(--border);'
        f'padding-top:8px;flex-wrap:wrap">{extra}</div></div>'
        f'<div class="tx-tile" style="gap:8px"><span class="lbl">Instances completed</span><div style="display:flex;'
        f'align-items:baseline;gap:8px;flex-wrap:wrap"><span class="tx-big">{done_tot:,}</span><span class="tx-ci">of '
        f'{exp_tot:,} across {len(plan)} runs · {ui.num(pct_done)}%</span></div>'
        f'<div role="img" aria-label="{pct_done:.0f} percent complete" style="height:4px;border-radius:2px;'
        f'background:var(--border);overflow:hidden;margin-top:8px"><div style="width:{pct_done}%;height:100%;'
        f'background:var(--accent)"></div></div><span class="tx-note">◐ {states["part"]} partial · ○ {states["none"]} '
        f'not run · about {ui.num(left_h, 1)} GPU-hours left</span></div></div>')

    ui.section("All runs", f"sample {sample}")
    st.html(ui.table([("Dataset", "l"), ("Model", "l"), ("Method", "l"), ("Names", "l"), ("Done / expected", "r"),
                      ("s / inst", "r"), ("Out tokens", "r"), ("GPU", "l"), ("Settings", "l"), ("Updated", "r")],
                     rows, min_width=1100))

    # Behaviour rates for runs that exist
    have = [(p, res["runs"][data.run_key(p, sample)]) for p in plan if data.run_key(p, sample) in res["runs"]]
    if not have:
        ui.empty_state("No runs in this sample yet", "Copy the Kaggle results zip into the project so the files land "
                       "in results/llm/&lt;dataset&gt;/, then refresh.")
        return

    def label(p, r):
        lab = f'{"MS Tokyo" if p["dataset"] == "msteps_tokyo" else ui.DATASET_SHORT[p["dataset"]]} · ' \
              f'{ui.METHOD_SHORT[p["method"]]} · {ui.MODEL_SHORT[p["model"]]}'
        if p["condition"] not in (C.NO_NAMES, "present"):
            lab += f" · {p['condition']}"
        return lab + (" ◐" if not r["complete"] else "")

    ui.section("Copy and invalid rates", ui.legend(
        [(ui.sw("sw-7"), "Qwen2.5-7B"), (ui.sw("sw-3"), "Qwen2.5-3B"),
         ('<span style="width:12px;height:1.5px;background:var(--ink)"></span>', "95% CI")]))
    copy_rows = [{"label": label(p, r), "v": 100 * r["copy"]["mean"], "lo": 100 * r["copy"]["lo"],
                  "hi": 100 * r["copy"]["hi"], "fill": ui.MODEL_VAR[p["model"]]} for p, r in have]
    inv_rows = [{"label": label(p, r), "v": 100 * r["invalid"]["mean"], "lo": 100 * r["invalid"]["lo"],
                 "hi": 100 * r["invalid"]["hi"], "fill": ui.MODEL_VAR[p["model"]]} for p, r in have]
    inv_max = max(10.0, max(x["hi"] for x in inv_rows))
    inv_max = 10 * -(-inv_max // 10)
    a, b = st.columns(2, gap="large")
    with a:
        st.html('<div class="tx-root tx-row" style="font-size:13px;font-weight:500"><span>Copy rate</span>'
                '<span class="tx-note" style="font-weight:400">top-1 is a place from the user’s history</span></div>'
                + ui.rate_bars(copy_rows, 100, ("0", "50", "100%")))
    with b:
        st.html('<div class="tx-root tx-row" style="font-size:13px;font-weight:500"><span>Invalid rate</span>'
                '<span class="tx-note" style="font-weight:400">no usable prediction</span></div>'
                + ui.rate_bars(inv_rows, inv_max, ("0", ui.num(inv_max / 2, 0), f"{ui.num(inv_max, 0)}%")))

    ui.section("Parse status", ui.legend([(ui.sw("var(--ink2)"), "JSON"), (ui.sw("var(--regex)"), "Regex fallback"),
                                          (ui.sw("var(--bad)"), "Failed")]))
    out = []
    for p, r in have:
        stt, n = r["status"], max(r["n"], 1)
        pj = 100 * (stt.get("json", 0) + stt.get("literal", 0)) / n
        pr = 100 * (stt.get("regex_list", 0) + stt.get("poi_mentions", 0)) / n
        pf = 100 * (stt.get("failed", 0) + stt.get("empty", 0)) / n
        out.append(f'<div class="hb-row stack-row"><span class="lab">{ui.esc(label(p, r))}</span>'
                   f'<span class="stack" role="img" aria-label="JSON {pj:.1f}%, regex {pr:.1f}%, failed {pf:.1f}%">'
                   f'<span style="background:var(--ink2);width:{pj}%"></span><span style="background:var(--regex);width:{pr}%">'
                   f'</span><span style="background:var(--bad);width:{pf}%"></span></span>'
                   f'<span class="tx-ci" style="text-align:right">{ui.num(pj)}% · {ui.num(pr)}% · {ui.num(pf)}%</span></div>')
    half = (len(out) + 1) // 2  # fill the left column first, then the right
    st.html('<div class="tx-root" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(480px,100%),1fr));'
            'column-gap:32px;row-gap:4px"><div>' + "".join(out[:half]) + "</div><div>" + "".join(out[half:]) + "</div></div>")
