"""Dataset configuration and the fixed experimental protocol for TraceX.

Every number that defines the protocol lives here so that all three members
(data, retrieval, LLM) work from exactly the same settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Protocol constants (shared by every dataset)
# ---------------------------------------------------------------------------
SESSION_GAP_HOURS = 24          # new trajectory when gap >= 24h (Nishida et al., LLM4POI)
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # chronological train / validation / test by check-in time
MIN_TRAJ_LEN = 2                # a trajectory needs >=1 input check-in plus the target
RANDOM_SEED = 42                # fixed seed for every sample
SAMPLE_SIZE = 500               # brief: about 300-500 test instances per setting
DEBUG_SAMPLE_SIZE = 20          # brief: debug on 20 instances before any full run
HISTORY_SPLITS = ("train",)     # fixed definition of "history" (see docs/data_pipeline.md)

FOURSQUARE_URL = "http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip"
MASSIVE_STEPS_REPO = "CRUISEResearchGroup/Massive-STEPS-{city}"
MASSIVE_STEPS_URL = "https://huggingface.co/datasets/" + MASSIVE_STEPS_REPO + "/resolve/main/{file}"


@dataclass(frozen=True)
class DatasetConfig:
    name: str                  # short key used for folders, e.g. "nyc"
    source: str                # "foursquare" or "massive_steps"
    raw_path: Path
    min_poi_checkins: int      # drop POIs with fewer check-ins (before splitting)
    min_user_checkins: int     # drop users with fewer check-ins (after POI filter)
    has_names: bool = False
    description: str = ""
    extra: dict = field(default_factory=dict)


DATASETS: dict[str, DatasetConfig] = {
    "nyc": DatasetConfig(
        name="nyc",
        source="foursquare",
        raw_path=RAW_DIR / "foursquare" / "dataset_tsmc2014" / "dataset_TSMC2014_NYC.txt",
        min_poi_checkins=10,
        min_user_checkins=10,
        description="Foursquare NYC (Yang et al. 2014), Apr 2012 - Feb 2013",
    ),
    "tky": DatasetConfig(
        name="tky",
        source="foursquare",
        raw_path=RAW_DIR / "foursquare" / "dataset_tsmc2014" / "dataset_TSMC2014_TKY.txt",
        min_poi_checkins=10,
        min_user_checkins=10,
        description="Foursquare TKY (Yang et al. 2014), Apr 2012 - Feb 2013",
    ),
    # Massive-STEPS is far sparser (median POI has 2-3 check-ins), so the
    # LLM4POI thresholds would delete most of it. We keep every POI and only
    # require users to have enough check-ins to form a train history plus a
    # test trajectory. E3 compares conditions *within* this dataset, so the
    # lighter filter does not affect the comparison.
    "msteps_tokyo": DatasetConfig(
        name="msteps_tokyo",
        source="massive_steps",
        raw_path=RAW_DIR / "massive_steps" / "tokyo" / "tokyo_checkins.csv",
        min_poi_checkins=1,
        min_user_checkins=5,
        has_names=True,
        description="Massive-STEPS Tokyo (Wongso et al.), Semantic Trails 2017-18, with venue names",
        extra={"city": "Tokyo", "timezone": "Asia/Tokyo"},
    ),
}
