"""Per-session context shared by all pages: selected sample, results signature, evaluation."""
from __future__ import annotations

import streamlit as st

import data
import ui


def sidebar():
    sig = data.signature()
    samples = [s for s in data.SAMPLES if data.has_results(s, sig)] or ["sample_test_500"]
    if st.session_state.get("sample") not in samples:
        st.session_state["sample"] = samples[0]
    with st.sidebar:
        st.html('<div class="tx-root" style="padding:0 2px;font-size:12px;line-height:16px;font-weight:600;'
                'color:var(--ink2)">Data</div>')
        st.selectbox("Sample", samples, key="sample")
        st.html(f'<div class="tx-root" style="padding:0 2px;display:flex;flex-direction:column;gap:12px">'
                f'<div><div class="tx-note">Results last updated</div>'
                f'<div class="tx-mono">{ui.esc(data.last_updated(sig))}</div></div>'
                f'<div class="tx-note">Light / dark: menu ⋮ at the top right</div></div>')


def get(loading_slot=None):
    """(sample, signature, evaluation). Shows a layout skeleton while results are first computed."""
    sample = st.session_state.get("sample", "sample_test_500")
    sig = data.signature()
    slot = loading_slot or st.empty()
    slot.html(ui.skeleton())
    res = data.evaluation(sample, sig)
    slot.empty()
    return sample, sig, res


def runs_for(res: dict, dataset=None, model=None, method=None, conditions=None) -> list[tuple[str, dict]]:
    out = []
    for key, r in res["runs"].items():
        if dataset and r["dataset"] != dataset:
            continue
        if model and r["model"] != model:
            continue
        if method and r["method"] != method:
            continue
        if conditions and r["condition"] not in conditions:
            continue
        out.append((key, r))
    return out
