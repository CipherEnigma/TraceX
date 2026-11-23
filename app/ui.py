"""Shared UI pieces for the TraceX dashboard: styles, formatting, HTML components.

Everything custom is plain HTML styled by tracex.css, so it shares the design tokens and
switches to dark mode with Streamlit's theme (the ⋮ menu, top right).
"""
from __future__ import annotations

import html
import math
from pathlib import Path

import streamlit as st

CSS = Path(__file__).with_name("tracex.css")

DARK_VARS = """:root[data-tx-theme="dark"]{--bg:#1A1A19;--surface:#232322;--border:#34332F;--ink:#F0EFEC;--ink2:#C3C2B7;--muted:#7A7973;
--mutedText:#A3A299;--accent:#6E9BD6;--accentTint:rgba(110,155,214,0.14);--s7:#3987E5;--s3:#D95926;--sa:#199E70;
--heur:#C3C2B7;--band:rgba(195,194,183,0.12);--ok:#5DB37E;--okTint:rgba(93,179,126,0.13);--bad:#E0735F;
--badTint:rgba(224,115,95,0.13);--hover:rgba(240,239,236,0.04);--nameTint:rgba(57,135,229,0.22);--skel:#2C2C2A;
--codeBg:#232322;--regex:#6B6A64;}"""

DATASETS = {"nyc": "Foursquare NYC", "tky": "Foursquare TKY", "msteps_tokyo": "Massive-STEPS Tokyo"}
DATASET_SHORT = {"nyc": "NYC", "tky": "TKY", "msteps_tokyo": "Massive-STEPS Tokyo"}
MODELS = {"qwen7b": "Qwen2.5-7B", "qwen3b": "Qwen2.5-3B"}
MODEL_SHORT = {"qwen7b": "7B", "qwen3b": "3B"}
MODEL_SW = {"qwen7b": "sw-7", "qwen3b": "sw-3"}
MODEL_VAR = {"qwen7b": "var(--s7)", "qwen3b": "var(--s3)"}
MODEL_HATCH = {"qwen7b": "hatch-7", "qwen3b": "hatch-3"}
METHODS = {"llmmob": "LLM-Mob", "retrieval-jaccard": "Retrieval (Jaccard)",
           "retrieval-lcs": "Retrieval (LCS)", "retrieval-bm25": "Retrieval (BM25)"}
METHOD_SHORT = {"llmmob": "LLM-Mob", "retrieval-jaccard": "Retrieval", "retrieval-lcs": "Retrieval (LCS)",
                "retrieval-bm25": "Retrieval (BM25)"}
CONDITIONS = {"present": "Names present", "removed": "Names removed", "aliased": "Names aliased"}
COND_VAR = {"present": "var(--s7)", "removed": "var(--s3)", "aliased": "var(--sa)"}
COND_SW = {"present": "sw-7", "removed": "sw-3", "aliased": "sw-a"}
COND_HATCH = {"present": "hatch-7", "removed": "hatch-3", "aliased": "hatch-a"}
MINUS = "−"


def esc(x) -> str:
    return html.escape(str(x), quote=True)


# ---------------------------------------------------------------------------
# Styles and page frame
# ---------------------------------------------------------------------------
# Streamlit does not tell the server when the viewer switches theme in the menu (streamlit#11920), so the
# browser decides: watch the app background Streamlit paints and set <html data-tx-theme="light|dark">,
# which switches the dark token block above. Runs once per page load; idempotent across reruns.
THEME_JS = """(function(){
  if (window.__txTheme) return; window.__txTheme = true;
  function sync(){
    var app = document.querySelector('.stApp'); if (!app) return;
    var c = (getComputedStyle(app).backgroundColor.match(/[\\d.]+/g) || []).map(Number);
    if (c.length < 3) return;
    var t = (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) < 128 ? 'dark' : 'light';
    if (document.documentElement.getAttribute('data-tx-theme') !== t) document.documentElement.setAttribute('data-tx-theme', t);
  }
  sync(); setInterval(sync, 250);
})();"""


def inject_css():
    css = CSS.read_text(encoding="utf-8")
    try:  # first paint before the script runs: use the server's guess
        dark = st.context.theme.type == "dark"
    except Exception:
        dark = False
    first = DARK_VARS.replace(':root[data-tx-theme="dark"]', ':root:not([data-tx-theme])') if dark else ""
    st.html(f"<style>{css}{DARK_VARS}{first}</style><script>{THEME_JS}</script>", unsafe_allow_javascript=True)


def page_header(title: str, desc: str):
    st.html(f'<div class="tx-root"><h1 class="tx-h1">{esc(title)}</h1><p class="tx-desc">{esc(desc)}</p></div>')


def rule():
    st.html('<div class="tx-rule"></div>')


def section(title: str, note: str = ""):
    st.html(f'<div class="tx-root tx-sechead"><h2 class="tx-h2">{esc(title)}</h2>'
            f'<span class="tx-note">{note}</span></div>')


def caption_html(text: str) -> str:
    return f'<p class="tx-cap">{text}</p>'


def caption(text: str):
    st.html(caption_html(text))


def segmented(label: str, options: dict, key: str, default=None):
    """st.segmented_control that always has a value. options: {value: label}."""
    keys = list(options)
    if key not in st.session_state or st.session_state[key] not in keys:
        st.session_state[key] = default if default in keys else keys[0]
    val = st.segmented_control(label, keys, format_func=lambda k: options[k], key=key)
    if val is None:  # clicking the selected segment deselects it in Streamlit; keep the last value
        val = st.session_state.get(f"_{key}_last", keys[0])
    st.session_state[f"_{key}_last"] = val
    return val


def select(label: str, options: dict, key: str, default=None, disabled=False):
    keys = list(options)
    idx = keys.index(default) if default in keys else 0
    return st.selectbox(label, keys, index=idx, format_func=lambda k: options[k], key=key, disabled=disabled)


# ---------------------------------------------------------------------------
# Number formatting (design: one decimal, true minus, signed deltas, p to 2 s.f.)
# ---------------------------------------------------------------------------
def num(v, digits=1, signed=False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    s = f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:] if not signed else "+" + s[1:]
    return s.replace("-", MINUS)


def pct(s: dict, scale=100.0) -> str:
    """'21.8 <ci>[18.2, 25.4]</ci>' from a {'mean','lo','hi'} summary."""
    if not s or s.get("n", 1) == 0:
        return "–"
    return (f"{num(scale * s['mean'])} <span class='tx-ci'>[{num(scale * s['lo'])}, "
            f"{num(scale * s['hi'])}]</span>")


def diff(d: dict, scale=100.0) -> str:
    if not d or d.get("n", 1) == 0:
        return "–"
    return (f"{num(scale * d['diff'], signed=True)} <span class='tx-ci'>[{num(scale * d['lo'], signed=True)}, "
            f"{num(scale * d['hi'], signed=True)}]</span>")


def pval(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    if p < 0.001:
        return "&lt;0.001"
    return f"{p:.2f}" if p >= 0.1 else f"{p:.2g}"


# ---------------------------------------------------------------------------
# Small components
# ---------------------------------------------------------------------------
def tag(text, kind=""):
    return f'<span class="tx-tag {kind}">{text}</span>'


def partial_tag():
    return tag("◐ partial", "partial")


def status(done: int, expected: int, applicable=True) -> str:
    if not applicable:
        return '<span class="tx-status st-na"><span class="ic">–</span>n/a</span>'
    if done >= expected > 0:
        return (f'<span class="tx-status"><span class="ic st-ok">✓</span>Complete'
                f'<span class="tx-ci">{done}/{expected}</span></span>')
    if done > 0:
        return (f'<span class="tx-status"><span class="ic st-run">◐</span>Partial'
                f'<span class="tx-ci">{done}/{expected}</span></span>')
    return '<span class="tx-status st-run"><span class="ic">○</span>Not run</span>'


def legend(items: list[tuple[str, str]]) -> str:
    """items: (swatch html, label)."""
    return '<div class="tx-legend">' + "".join(f"<span>{s}{l}</span>" for s, l in items) + "</div>"


def sw(cls_or_var: str, big=True) -> str:
    if cls_or_var.startswith("var("):
        return f'<span class="sw {"sw-lg" if big else ""}" style="background:{cls_or_var}"></span>'
    return f'<span class="sw {"sw-lg" if big else ""} {cls_or_var}"></span>'


def table(columns: list[tuple[str, str]], rows: list[dict], min_width=0) -> str:
    """columns: (label, 'l'|'r'); rows: {'cells': [...html], 'cls': ''}."""
    head = "".join(f'<th class="{"r" if a == "r" else ""}" scope="col">{esc(c)}</th>' for c, a in columns)
    body = []
    for r in rows:
        tds = "".join(f'<td class="{"r" if a == "r" else ""} {extra}">{cell}</td>'
                      for (cell, extra), (_, a) in zip(
                          [(c if isinstance(c, tuple) else (c, "")) for c in r["cells"]], columns))
        body.append(f'<tr class="{r.get("cls", "")}">{tds}</tr>')
    mw = f' style="min-width:{min_width}px"' if min_width else ""
    return (f'<div class="tx-tablewrap"><table class="tx-table"{mw}><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def empty_state(title: str, body: str, command: str | None = None, extra: str = "", status_html: str = ""):
    cmd = f'<pre class="tx-code">{esc(command)}</pre>' if command else ""
    st.html(f'<div class="tx-root tx-empty">{status_html}<h2>{esc(title)}</h2><p>{body}</p>{cmd}{extra}</div>')


def skeleton(rows=3) -> str:
    bars = "".join(f'<div class="sk-row"><span class="sk" style="height:20px;width:{w}%"></span></div>'
                   for w in (60, 85, 40, 75, 55)[:rows])
    return (f'<div class="sk-wrap" aria-busy="true"><span class="tx-note">Loading results…</span>'
            f'<div class="sk-row"><span class="sk" style="height:96px;flex:1"></span>'
            f'<span class="sk" style="height:96px;flex:1"></span><span class="sk" style="height:96px;flex:1"></span></div>'
            f'<span class="sk" style="height:220px"></span>{bars}</div>')


# ---------------------------------------------------------------------------
# Vertical grouped bar chart (HTML)
# ---------------------------------------------------------------------------
def nice_max(vmax: float) -> tuple[float, float]:
    """Axis maximum and tick step (percent units)."""
    vmax = max(vmax, 1.0)
    for step in (5, 10, 20, 25):
        if vmax / step <= 5:
            return step * math.ceil(vmax / step + 1e-9), step
    return 25 * math.ceil(vmax / 25), 25


def bar_chart(groups: list[dict], heuristic: dict | None = None, height=300, bar_w=72, gap=16,
              label_mode="value", aria="") -> str:
    """groups: [{'label', 'bars': [{'v','lo','hi','fill' (css var or class), 'hatch': bool,
    'sub': x label, 'subnote': small note, 'tip': [lines], 'big': (value, note) or None}]}].
    Values are in percent. label_mode: 'value' (all bars), 'none'."""
    vals = [b["hi"] for g in groups for b in g["bars"] if b.get("hi") is not None]
    if heuristic:
        vals.append(heuristic["hi"])
    ymax, step = nice_max(max(vals) * 1.08 if vals else 10)
    P = lambda v: max(0.0, min(100.0, 100.0 * v / ymax))  # noqa: E731
    ticks = [i * step for i in range(int(round(ymax / step)) + 1)]
    axis = "".join(
        f'<span style="bottom:calc({P(t)}% - 8px)">{num(t, 0)}{"%" if t == ticks[-1] else ""}</span>' for t in ticks)
    grid = "".join(f'<div class="vc-grid" style="bottom:{P(t)}%"></div>' for t in ticks[1:])
    heur = ""
    if heuristic:
        heur = (f'<div class="vc-band" style="bottom:{P(heuristic["lo"])}%;height:{P(heuristic["hi"]) - P(heuristic["lo"])}%"></div>'
                f'<div class="vc-heur" style="bottom:{P(heuristic["v"])}%"></div>'
                f'<span class="vc-heurlab" style="bottom:calc({P(heuristic["hi"])}% + 4px)">Heuristic {num(heuristic["v"])}</span>')
    gh, xl = [], []
    for g in groups:
        bars, subs = [], []
        for b in g["bars"]:
            fill = b["fill"]
            if b.get("hatch"):
                fill_html = f'<div class="vc-fill {b["hatch_cls"]}" style="height:{P(b["v"])}%"></div>'
            else:
                fill_html = f'<div class="vc-fill" style="height:{P(b["v"])}%;background:{fill}"></div>'
            wh = max(P(b["hi"]) - P(b["lo"]), 0.0)
            whisk = f'<div class="vc-whisk" style="bottom:{P(b["lo"])}%;height:{wh}%"></div>'
            if b.get("big"):
                big_v, big_n = b["big"]
                lab = (f'<span class="vc-val big" style="bottom:calc({P(b["hi"])}% + 6px)"><b>{num(big_v)}</b>'
                       f'<i>{esc(big_n)}</i></span>')
            elif label_mode == "value":
                lab = f'<span class="vc-val" style="bottom:calc({P(b["hi"])}% + 4px)"><span>{num(b["v"])}</span></span>'
            else:
                lab = ""
            tip = ""
            if b.get("tip"):
                tip = f'<div class="vc-tip"><b>{b["tip"][0]}</b>' + "".join(
                    f"<span>{t}</span>" for t in b["tip"][1:]) + "</div>"
            bars.append(f'<div class="vc-bar" style="width:{bar_w}px">{fill_html}{whisk}{lab}{tip}</div>')
            note = f'<small>{esc(b["subnote"])}</small>' if b.get("subnote") else ""
            subs.append(f'<span style="width:{bar_w}px">{esc(b.get("sub", ""))}{note}</span>')
        gw = len(g["bars"]) * bar_w + (len(g["bars"]) - 1) * gap
        gh.append(f'<div class="vc-group" style="gap:{gap}px">{"".join(bars)}</div>')
        xl.append(f'<div class="vc-xgroup" style="width:{gw}px"><div class="vc-subs" style="gap:{gap}px">'
                  f'{"".join(subs)}</div><div class="vc-glab">{esc(g["label"])}</div></div>')
    return (f'<div class="vc tx-root"><div class="vc-axis" style="height:{height}px" aria-hidden="true">{axis}</div>'
            f'<div class="vc-body"><div class="vc-plot" role="img" aria-label="{esc(aria)}" style="height:{height}px">'
            f'{grid}{heur}<div class="vc-groups">{"".join(gh)}</div></div>'
            f'<div class="vc-xlabs">{"".join(xl)}</div></div></div>')


# ---------------------------------------------------------------------------
# Horizontal rate bars with CI (Runs page)
# ---------------------------------------------------------------------------
def rate_bars(rows: list[dict], axis_max: float, axis_labels: tuple[str, str, str]) -> str:
    """rows: {'label','v','lo','hi','fill'} in percent."""
    P = lambda v: max(0.0, min(100.0, 100.0 * v / axis_max))  # noqa: E731
    out = []
    for r in rows:
        out.append(
            f'<div class="hb-row"><span class="lab" title="{esc(r["label"])}">{esc(r["label"])}</span>'
            f'<span class="hb-track"><span class="hb-fill" style="width:{P(r["v"])}%;background:{r["fill"]}"></span>'
            f'<span class="hb-ci" style="left:{P(r["lo"])}%;width:{max(P(r["hi"]) - P(r["lo"]), 0.4)}%"></span></span>'
            f'<span class="val">{num(r["v"])}</span></div>')
    a, b, c = axis_labels
    out.append(f'<div class="hb-axis"><span></span><span><span>{a}</span><span>{b}</span><span>{c}</span></span><span></span></div>')
    return '<div class="tx-root">' + "".join(out) + "</div>"
