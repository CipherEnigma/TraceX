"""Fixed settings for every LLM run in TraceX (Member 3).

All LLM numbers in the paper come from these settings, so change them here only,
never inside a script. The data protocol (splits, history, samples, seed) is
imported from `tracex_data.config` so that every method sees the same instances.
"""
from __future__ import annotations

from dataclasses import dataclass

from tracex_data.config import RANDOM_SEED, REPORTS_DIR, ROOT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = ROOT / "results" / "llm"     # raw per-instance JSONL outputs (one file per run)
LLM_REPORTS_DIR = REPORTS_DIR / "llm"      # tables and figures for E1-E3
PERTURBED_DIR = ROOT / "data" / "perturbed"  # E3 name-perturbed test sets (release deliverable)

SEED = RANDOM_SEED  # 42, shared with the data pipeline


# ---------------------------------------------------------------------------
# Models (brief: Qwen2.5-7B-Instruct 4-bit, plus Qwen2.5-3B as a size comparison)
# Both run on the same Kaggle GPU so that results and timings are comparable.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    key: str                 # short name used in result file names
    hf_id: str | None        # Hugging Face model id (None for the mock backend)
    backend: str             # "hf" (transformers + bitsandbytes) or "mock"
    load_in_4bit: bool = True


MODELS: dict[str, ModelConfig] = {
    "qwen7b": ModelConfig("qwen7b", "Qwen/Qwen2.5-7B-Instruct", "hf"),
    "qwen3b": ModelConfig("qwen3b", "Qwen/Qwen2.5-3B-Instruct", "hf"),
    # Deterministic stand-in for tests and the dashboard; never used for reported numbers.
    "mock": ModelConfig("mock", None, "mock", load_in_4bit=False),
}

# 4-bit NF4 quantisation. T4 GPUs have no bfloat16 support, so compute in float16.
QUANTIZATION = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "float16",
}

# Brief: temperature 0 -> greedy decoding, no sampling.
GENERATION = {
    "do_sample": False,
    "max_new_tokens": 256,
}


# ---------------------------------------------------------------------------
# Prompting methods
# ---------------------------------------------------------------------------
METHODS = ("llmmob", "retrieval")
# llmmob    : LLM-Mob style - the user's most recent history check-ins + current prefix.
# retrieval : the user's past trajectories most similar to the prefix (Nishida et al.),
#             selected by Member 2's `tracex_ir.retrieval.retrieve_demonstrations`.

N_PREDICTIONS = 10            # ranked POI ids requested from the model (>= 5 for Acc@5 / NDCG@5)
LLMMOB_MAX_HISTORY = 40       # most recent history check-ins shown (LLM-Mob uses 40 historical stays)
RETRIEVERS = ("jaccard", "lcs", "bm25")
DEFAULT_RETRIEVER = "jaccard" # brief names Jaccard / LCS selection for the second LLM method
DEMO_TOP_K = 5                # demonstrations per prompt (low end of Nishida et al.'s 5-30)
DEMO_MAX_CONTEXT = 20         # most recent check-ins kept in each demonstration's context
PREFIX_MAX_CHECKINS = 30      # most recent check-ins of the current trajectory shown

# The prompt states *when* the next check-in happens (LLM-Mob, LLM4POI and Nishida et al.
# all do this). Only the target's `time` and `weekday` are read - never its POI, category
# or name. Set to False for a stricter no-target-information variant.
INCLUDE_TARGET_TIME = True


# ---------------------------------------------------------------------------
# E3 name test (Massive-STEPS only; NYC and TKY have no venue names)
# ---------------------------------------------------------------------------
NAME_DATASETS = ("msteps_tokyo",)
NAME_CONDITIONS = ("present", "removed", "aliased")
NO_NAMES = "none"             # condition tag for datasets without names


def name_conditions(dataset: str) -> tuple[str, ...]:
    return NAME_CONDITIONS if dataset in NAME_DATASETS else (NO_NAMES,)


def run_name(dataset: str, model: str, method: str, condition: str = NO_NAMES,
             retriever: str | None = None, sample: str = "sample_test_500") -> str:
    """Canonical run id, also the result file stem, e.g. `nyc__qwen7b__retrieval-jaccard__none__sample_test_500`."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {tuple(MODELS)}")
    if condition not in name_conditions(dataset):
        raise ValueError(f"condition {condition!r} not valid for {dataset}; expected {name_conditions(dataset)}")
    m = method
    if method == "retrieval":
        r = retriever or DEFAULT_RETRIEVER
        if r not in RETRIEVERS:
            raise ValueError(f"unknown retriever {r!r}; expected one of {RETRIEVERS}")
        m = f"retrieval-{r}"
    return f"{dataset}__{model}__{m}__{condition}__{sample}"


def result_path(dataset: str, model: str, method: str, condition: str = NO_NAMES,
                retriever: str | None = None, sample: str = "sample_test_500"):
    return RESULTS_DIR / dataset / f"{run_name(dataset, model, method, condition, retriever, sample)}.jsonl"
