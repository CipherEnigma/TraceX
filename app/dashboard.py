"""TraceX dashboard (Member 3).

    pip install -r requirements-llm.txt
    streamlit run app/dashboard.py

Reads results/llm/ (copy Kaggle results there) and the processed data in data/processed/.
Set TRACEX_RESULTS_DIR to read results from another folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import os  # noqa: E402

import streamlit as st  # noqa: E402

from tracex_llm import config as C  # noqa: E402

if os.environ.get("TRACEX_RESULTS_DIR"):  # read results from elsewhere (demos, tests)
    C.RESULTS_DIR = Path(os.environ["TRACEX_RESULTS_DIR"]).resolve()

import ui  # noqa: E402
from views import e1, e2, e3, explorer, overview, runs  # noqa: E402

st.set_page_config(page_title="TraceX", page_icon=":material/insights:", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

pg = st.navigation([
    st.Page(overview.render, title="Overview", icon=":material/grid_view:", default=True),
    st.Page(e1.render, title="E1 Heuristic floor", icon=":material/bar_chart:", url_path="e1"),
    st.Page(e2.render, title="E2 Revisit vs new", icon=":material/sync:", url_path="e2"),
    st.Page(e3.render, title="E3 Name test", icon=":material/sell:", url_path="e3"),
    st.Page(explorer.render, title="Instance explorer", icon=":material/search:", url_path="explorer"),
    st.Page(runs.render, title="Runs & behaviour", icon=":material/monitoring:", url_path="runs"),
])

import context  # noqa: E402

context.sidebar()
pg.run()
