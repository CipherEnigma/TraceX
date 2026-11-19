"""CLI:  python -m tracex_data {download,run,figures,report,all} [--dataset nyc tky msteps_tokyo]

    download  fetch raw data into data/raw/
    run       preprocess the chosen datasets, then rebuild figures and report
    figures   rebuild figures only
    report    rebuild reports/data_report.md only
    all       download + run
"""
from __future__ import annotations

import argparse

from .config import DATASETS


def main() -> None:
    p = argparse.ArgumentParser(prog="tracex_data")
    p.add_argument("command", choices=["download", "run", "figures", "report", "all"])
    p.add_argument("--dataset", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    a = p.parse_args()

    if a.command in ("download", "all"):
        from .download import download_all
        download_all()
    if a.command in ("run", "all"):
        from .pipeline import process_many
        process_many(a.dataset)
    if a.command in ("figures", "run", "all"):
        from .plots import make_figures
        for f in make_figures():
            print(f"[figures] {f}")
    if a.command in ("report", "run", "all"):
        from .report import write_report
        print(f"[report] {write_report()}")


if __name__ == "__main__":
    main()
