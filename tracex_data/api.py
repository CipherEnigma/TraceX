"""Loader API for the retrieval (Member 2) and LLM (Member 3) code.

    from tracex_data.api import load_dataset
    ds = load_dataset("nyc")
    for inst in ds.instances("sample_test_500"):
        history = ds.history(inst)        # list of earlier trajectories (dicts)
        prefix  = inst["prefix"]          # current trajectory without the target
        target  = inst["target"]["poi_id"]  # ground truth - never put in a prompt
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property

import pandas as pd

from .config import PROCESSED_DIR


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class ProcessedDataset:
    name: str
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def root(self):
        return PROCESSED_DIR / self.name

    @cached_property
    def trajectories(self) -> dict[int, dict]:
        return {t["traj_id"]: t for t in read_jsonl(self.root / "trajectories.jsonl")}

    @cached_property
    def pois(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "pois.csv")

    @cached_property
    def stats(self) -> dict:
        return json.loads((self.root / "stats.json").read_text(encoding="utf-8"))

    def checkins(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "checkins.csv", parse_dates=["local_time"])

    def instances(self, which: str = "sample_test_500") -> list[dict]:
        """which: 'val', 'test', 'sample_test_500' or 'sample_test_debug20'."""
        fname = f"instances_{which}.jsonl" if which in ("val", "test") else f"{which}.jsonl"
        if fname not in self._cache:
            self._cache[fname] = read_jsonl(self.root / fname)
        return self._cache[fname]

    def history(self, inst: dict) -> list[dict]:
        """User's earlier training trajectories, oldest first (the fixed history definition)."""
        return [self.trajectories[i] for i in inst["history_traj_ids"]]

    def train_trajectories(self, user_id: int | None = None) -> list[dict]:
        return [t for t in self.trajectories.values()
                if t["split"] == "train" and (user_id is None or t["user_id"] == user_id)]


def load_dataset(name: str) -> ProcessedDataset:
    if not (PROCESSED_DIR / name / "stats.json").exists():
        raise FileNotFoundError(f"{name} not processed - run `python -m tracex_data run --dataset {name}`")
    return ProcessedDataset(name)
