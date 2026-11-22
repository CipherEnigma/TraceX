"""Build report/report.html and report/TraceX_report.pdf from the result files.

    python report/build_report.py

Tables are generated from reports/llm/results.md, reports/retrieval/*.json and tracex_ir/defaults.json,
so the numbers in the paper always match the saved results.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
DS = [("nyc", "Foursquare NYC"), ("tky", "Foursquare TKY"), ("msteps_tokyo", "Massive-STEPS Tokyo")]
AUTHORS = "Anushi [surname] &nbsp;&middot;&nbsp; Pragunie Aditya &nbsp;&middot;&nbsp; Aayush Gupta"
MODEL = {"qwen7b": "Qwen2.5-7B", "qwen3b": "Qwen2.5-3B"}
MODEL_S = {"qwen7b": "7B", "qwen3b": "3B"}
METHOD = {"llmmob": "LLM-Mob", "retrieval-jaccard": "Retrieval (Jaccard)"}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def md_section(text, heading):
    """Rows of the markdown tables under `## heading` (header and separator rows dropped)."""
    block = text.split(f"## {heading}")[1].split("\n## ")[0]
    rows = []
    for line in block.splitlines():
        if line.startswith("|") and not re.match(r"^\|[-| ]+\|$", line):
            rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def table(head, rows, bold_rows=(), sep_rows=(), left=1):
    h = "".join(f"<th>{c}</th>" for c in head)
    body = []
    for i, r in enumerate(rows):
        c = []
        if i in bold_rows:
            c.append("b")
        if i in sep_rows:
            c.append("sep")
        body.append(f'<tr class="{" ".join(c)}">' + "".join(f"<td>{x}</td>" for x in r) + "</tr>")
    return f'<table class="L{left}">\n<tr>{h}</tr>\n' + "\n".join(body) + "\n</table>"


def short_ci(s):
    """'23.2 [19.6, 27.0]' -> '23.2 [19.6, 27.0]' with a thin separator."""
    return s.replace("[", "[").replace(", ", ", ")


# --------------------------------------------------------------------------- data table
def table_data():
    head = ["", "Foursquare NYC", "Foursquare TKY", "Massive-STEPS Tokyo"]
    rows = [
        ["Period", "Apr 2012&ndash;Feb 2013", "Apr 2012&ndash;Feb 2013", "Oct 2017&ndash;Oct 2018"],
        ["Users", "1,073", "2,281", "719"],
        ["POIs", "5,061", "7,829", "2,517"],
        ["Categories", "208", "188", "275"],
        ["Check-ins", "122,783", "402,316", "12,184"],
        ["Trajectories", "23,649", "65,604", "4,416"],
        ["Check-ins per user (median)", "67", "110", "10"],
        ["Trajectory length (mean / median)", "5.19 / 3", "6.13 / 4", "2.76 / 2"],
        ["Test instances (all)", "2,386", "6,976", "334"],
        ["Revisit share, all test instances", "73.7%", "73.5%", "47.3%"],
        ["Evaluation sample (instances)", "500", "500", "334"],
        ["Revisit share, evaluation sample", "78.2%", "72.2%", "47.3%"],
        ["Venue names", "no", "no", "yes"],
    ]
    return table(head, rows)


# --------------------------------------------------------------------------- retrieval tables
def tables_retrieval():
    out = []
    n = 2
    for key, name in DS:
        r = json.load(open(ROOT / "reports/retrieval" / f"{key}_results.json"))
        head = ["Method", "P@1", "P@3", "R@3", "F1@3", "R@5", "F1@5", "MAP", "MRR"]
        names = {"recent": "Recent (no similarity)", "jaccard": "Jaccard", "lcs": "LCS", "tfidf": "TF-IDF cosine",
                 "bm25": "BM25", "hybrid": "Hybrid (RRF)"}
        rows = []
        for m, v in r["retrieval"].items():
            rows.append([names[m]] + [f"{v[k]:.3f}" for k in ["P@1", "P@3", "R@3", "F1@3", "R@5", "F1@5", "MAP", "MRR"]])
        best = max(range(len(rows)), key=lambda i: float(rows[i][7]))
        out.append(f'<div class="tcap"><b>Table {n}.</b> Retrieval quality, {name} ({r["n_instances"]} instances; '
                   f'coverage {r["coverage"]*100:.0f}%, scored instances only). Best MAP in bold.</div>')
        out.append(table(head, rows, bold_rows=(best,)))
        n += 1
    return "\n".join(out)


def table_val():
    d = json.load(open(ROOT / "tracex_ir/defaults.json"))
    m = d["val_mean_map"]
    rows = [[f"BM25 (k<sub>1</sub> = 1.5, b = {b})", f"{m[f'bm25(k1=1.5,b={b})']:.3f}"] for b in ["0.0", "0.25", "0.5", "0.75", "1.0"]]
    rows += [["Jaccard", f"{m['jaccard']:.3f}"], ["LCS", f"{m['lcs']:.3f}"], ["TF-IDF cosine", f"{m['tfidf']:.3f}"],
             ["Hybrid (BM25 + LCS + Jaccard)", f"{m['hybrid(bm25+lcs+jaccard)']:.3f}"],
             ["Hybrid (BM25 + LCS)", f"{m['hybrid(bm25+lcs)']:.3f}"]]
    return ('<div class="tcap"><b>Table 5.</b> Validation MAP (mean over the three datasets, 500 validation instances each). '
            'The best row is the setting used for the test results.</div>'
            + table(["Retriever / setting", "Mean validation MAP"], rows, bold_rows=(0,), sep_rows=(5,)))


# --------------------------------------------------------------------------- LLM tables
PLAIN = {  # plain visit counter, from the first full evaluation (before the time-aware heuristic replaced it)
    "nyc": ("21.8 [18.2, 25.4]", "57.8", "0.374", "0.412"),
    "tky": ("19.2 [15.8, 22.6]", "43.0", "0.292", "0.314"),
    "msteps_tokyo": ("16.8 [12.9, 21.0]", "38.9", "0.265", "0.286"),
}


def table_e1(md):
    rows_md = md_section(md, "E1: heuristic floor")
    rows, bold = [], []
    for key, name in DS:
        first = True
        group = [r for r in rows_md if r[0] == key]
        order = [r for r in group if r[1] == "frequency"]
        for model in ("qwen7b", "qwen3b"):
            for meth in ("llmmob", "retrieval-jaccard"):
                order += [r for r in group if r[2] == model and r[1] == meth]
        # plain counter row after the heuristic
        for r in order:
            label_ds = name if first else ""
            if r[1] == "frequency":
                rows.append([label_ds, "Time-aware heuristic", "&ndash;", r[4], r[5].split(" ")[0], r[6].split(" ")[0], r[7].split(" ")[0], "&ndash;", "&ndash;"])
                bold.append(len(rows) - 1)
                p = PLAIN[key]
                rows.append(["", "Plain visit counter", "&ndash;", p[0], p[1], p[2], p[3], "&ndash;", "&ndash;"])
            else:
                rows.append([label_ds, METHOD[r[1]], MODEL_S[r[2]], r[4], r[5].split(" ")[0], r[6].split(" ")[0], r[7].split(" ")[0], r[10].split(" ")[0], r[11]])
            first = False
    seps = tuple(i for i, r in enumerate(rows) if r[0] and i > 0)
    head = ["Dataset", "Method", "Qwen2.5", "Acc@1 [95% CI]", "Acc@5", "MRR", "NDCG@5", "&Delta; Acc@1", "<i>p</i>"]
    # drop the CI brackets of the delta column for width
    return table(head, rows, bold_rows=tuple(bold), sep_rows=seps, left=3)


def table_e2(md):
    rows_md = md_section(md, "E2: revisit vs new targets (Acc@1)")
    rows = []
    for key, name in DS:
        first = True
        for model in ("qwen7b", "qwen3b"):
            for meth in ("llmmob", "retrieval-jaccard"):
                rev = [r for r in rows_md if r[0] == key and r[1] == meth and r[2] == model and r[3] == "revisit"][0]
                new = [r for r in rows_md if r[0] == key and r[1] == meth and r[2] == model and r[3] == "new"][0]
                rows.append([name if first else "", METHOD[meth], MODEL_S[model], rev[4], rev[5].split(" ")[0],
                             rev[6].split(" ")[0], rev[7].split(" [")[0], rev[8], new[4], new[5].split(" ")[0],
                             new[6].split(" ")[0]])
                first = False
    seps = tuple(i for i, r in enumerate(rows) if r[0] and i > 0)
    head = ["Dataset", "Method", "Qwen2.5", "n rev.", "LLM", "Heur.", "&Delta;", "<i>p</i>", "n new", "LLM", "Heur."]
    return table(head, rows, sep_rows=seps, left=3)


def table_e3(md):
    rows_md = md_section(md, "E3: venue-name test (Massive-STEPS)")
    main = [r for r in rows_md if len(r) == 9]
    rows = []
    for model in ("qwen7b", "qwen3b"):
        for meth in ("llmmob", "retrieval-jaccard"):
            first = True
            for cond in ("present", "removed", "aliased"):
                r = [x for x in main if x[0] == model and x[1] == meth and x[2] == cond][0]
                rows.append([f"{MODEL[model]}, {METHOD[meth]}" if first else "", cond, r[4], r[5], r[6].split(" ")[0], r[7]])
                first = False
    seps = tuple(i for i, r in enumerate(rows) if r[0] and i > 0)
    head = ["Model, prompt", "Names", "Acc@1 [95% CI]", "Acc@5 [95% CI]", "MRR", "Copy"]
    return table(head, rows, sep_rows=seps, left=2)


def table_size(md):
    rows_md = md_section(md, "Model size: 7B − 3B")
    rows = []
    for r in rows_md[1:] if rows_md[0][0] == "Dataset" else rows_md:
        ds = dict(DS)[r[0]]
        cond = "" if r[2] == "none" else r[2]
        rows.append([ds, METHOD[r[1]], cond or "&ndash;", r[4], r[6]])
    return table(["Dataset", "Prompt", "Names", "&Delta; Acc@1 [95% CI]", "<i>p</i>"], rows, left=3)


def table_base():
    rows = []
    names = {"frequency": "Time-aware heuristic", "plain_frequency": "Plain visit counter",
             "recent_vote": "Votes: recent", "jaccard_vote": "Votes: Jaccard", "lcs_vote": "Votes: LCS",
             "tfidf_vote": "Votes: TF-IDF", "bm25_vote": "Votes: BM25", "hybrid_vote": "Votes: hybrid"}
    for key, name in DS:
        r = json.load(open(ROOT / "reports/retrieval" / f"{key}_results.json"))
        first = True
        for m, label in names.items():
            v = r["prediction"][m]
            rows.append([name if first else "", label, f"{v['acc1']*100:.1f}", f"{v['acc5']*100:.1f}",
                         f"{v['mrr']:.3f}", f"{v['ndcg5']:.3f}"])
            first = False
    seps = tuple(i for i, r in enumerate(rows) if r[0] and i > 0)
    bold = tuple(i for i, r in enumerate(rows) if r[1] == "Time-aware heuristic")
    return table(["Dataset", "Method", "Acc@1", "Acc@5", "MRR", "NDCG@5"], rows, bold_rows=bold, sep_rows=seps, left=2)


# --------------------------------------------------------------------------- diagram
def box(x, y, w, h, title, lines, fill, stroke):
    t = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
    t += f'<text x="{x + w/2}" y="{y + 15}" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#111">{title}</text>'
    for i, l in enumerate(lines):
        t += f'<text x="{x + w/2}" y="{y + 29 + i*12}" text-anchor="middle" font-size="8.8" fill="#222">{l}</text>'
    return t


def arrow(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/>')


def diagram():
    B, G, O = ("#e6eefb", "#2f5fa8"), ("#e3f3e6", "#2f7d3c"), ("#fdecd9", "#b5651d")
    s = ['<svg viewBox="0 0 700 330" width="100%" font-family="Times New Roman, serif">',
         '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
         '<path d="M0,0 L8,4 L0,8 z" fill="#333"/></marker></defs>']
    s.append(box(10, 10, 150, 66, "Raw check-ins", ["Foursquare NYC, TKY", "Massive-STEPS Tokyo"], *B))
    s.append(box(190, 10, 210, 66, "Preprocessing (data stage)", ["clean, filter, 80/10/10 time split,", "24 h trajectories, leakage checks"], *B))
    s.append(box(430, 10, 260, 66, "Prediction instances", ["prefix + history  &rarr;  target (last check-in)", "revisit / new label (scoring only)"], *B))
    s += [arrow(160, 43, 190, 43), arrow(400, 43, 430, 43)]
    # three predictors
    s.append(box(20, 128, 200, 78, "Frequency heuristic", ["POI count, recency decay,", "hour-of-day bonus", "(no LLM)"], *G))
    s.append(box(250, 128, 200, 78, "Retrieval (IR stage)", ["Jaccard, LCS, TF-IDF, BM25, hybrid", "query = prefix, pool = history", "top-k similar trajectories"], *G))
    s.append(box(480, 128, 200, 78, "LLM-Mob prompt", ["40 most recent history check-ins", "+ current trajectory + time", "Qwen2.5-7B / 3B, 4-bit"], *O))
    s.append(box(250, 236, 200, 60, "Retrieval prompt", ["5 Jaccard demonstrations", "+ current trajectory + time"], *O))
    s += [arrow(560, 76, 560, 128), arrow(350, 76, 350, 128), arrow(120, 76, 120, 128)]
    s += [arrow(350, 206, 350, 236)]
    s.append(box(480, 236, 200, 60, "Names (E3 only)", ["present / removed / aliased", "on Massive-STEPS Tokyo"], *O))
    s += [arrow(480, 266, 450, 266)]
    s.append(box(20, 236, 200, 60, "Scoring", ["Acc@1/5, MRR, NDCG@5, P/R/F1@K, MAP", "bootstrap CI, McNemar, Holm"], *B))
    s += [arrow(120, 206, 120, 236), arrow(250, 266, 220, 266)]
    s.append('<text x="350" y="318" text-anchor="middle" font-size="9" fill="#333">E1 compares all predictors, E2 splits by revisit / new, E3 varies the names in the prompts</text>')
    s.append("</svg>")
    return "\n".join(s)


def main():
    md = (ROOT / "reports/llm/results.md").read_text(encoding="utf-8")
    html = (OUT / "template.html").read_text(encoding="utf-8")
    parts = {"AUTHORS": AUTHORS, "DIAGRAM": diagram(), "TABLE_DATA": table_data(),
             "TABLES_RETRIEVAL": tables_retrieval(), "TABLE_VAL": table_val(), "TABLE_E1": table_e1(md),
             "TABLE_E2": table_e2(md), "TABLE_E3": table_e3(md), "TABLE_SIZE": table_size(md), "TABLE_BASE": table_base()}
    for k, v in parts.items():
        html = html.replace("{{" + k + "}}", v)
    assert "{{" not in html, re.findall(r"\{\{[A-Z_]+\}\}", html)
    (OUT / "report.html").write_text(html, encoding="utf-8")
    print("wrote report.html")


if __name__ == "__main__":
    main()
