# TraceX: Copy, Recall, or Reason?

Do LLMs predict a user's next point of interest by reasoning, or by copying the user's habits and recalling memorised venue knowledge? TraceX compares a frequency heuristic with two LLM prompting methods under one protocol, then runs two diagnostic tests: revisit versus new, and venue-name disguise.

## Data pipeline (Member 1)

```bash
pip install -r requirements.txt
python -m tracex_data all      # download + preprocess NYC, TKY, Massive-STEPS Tokyo + figures + report
python -m pytest tests -q
```

- Protocol, file formats and the loader API: [docs/data_pipeline.md](docs/data_pipeline.md)
- Statistics, cleaning log and leakage checks: [reports/data_report.md](reports/data_report.md)
- Massive-STEPS city choice: [reports/massive_steps_city_profile.md](reports/massive_steps_city_profile.md)

## Retrieval and baselines (Member 2)

```bash
python -m tracex_ir.evaluate                 # all datasets -> reports/retrieval/*.json and *.png
python -m pytest tests/test_ir.py -q
```

- `tracex_ir/retrieval.py`: Jaccard, LCS, TF-IDF cosine, BM25, hybrid (reciprocal rank fusion), plus `retrieve_demonstrations(inst, history, idf, method=None, top_k=None)`, the demonstration format for Member 3.
- `tracex_ir/baselines.py`: frequency baseline and retrieval-vote baselines. `tracex_ir/metrics.py`: P@K, R@K, MAP, MRR, Acc@K, NDCG@K.

### Interface for Member 3 (LLM)

```python
from tracex_data.api import load_dataset
from tracex_ir.retrieval import IDF, retrieve_demonstrations

ds = load_dataset("nyc")
idf = IDF(ds.train_trajectories())            # build once per dataset
for inst in ds.instances("sample_test_500"):
    demos = retrieve_demonstrations(inst, ds.history(inst), idf)   # top-3 BM25 by default
    # each demo: {"traj_id", "score", "context": [check-in, ...], "next": check-in, "checkins": [...]}
    # prompt idea: show demo["context"] then demo["next"] as the known answer, then ask for inst["prefix"]'s next POI
```

- The query uses only `inst["prefix"]`; the target is never read. Candidates are the user's history trajectories.
- Check-ins are the dicts from the data pipeline (`poi_id`, `category`, `time`, `hour`, `weekday`, plus `name` for Massive-STEPS).
- Default method and top-k are in `tracex_ir/defaults.json`, selected on the validation split with `python -m tracex_ir.tune` (BM25, k1=1.5, b=0, top-k 3).
- E1 non-LLM reference numbers: `reports/retrieval/<dataset>_results.json`, keys `prediction.frequency` and `prediction.<method>_vote`.

## LLM pipeline, experiments and dashboard (Member 3)

```bash
pip install -r requirements-llm.txt
python -m tracex_llm.run --dataset nyc --model qwen7b --sample sample_test_debug20   # needs a CUDA GPU
python -m tracex_llm.evaluate && python -m tracex_llm.report                        # tables + figures
streamlit run app/dashboard.py                                                       # dashboard
python -m pytest tests/test_llm.py -q                                                # no GPU needed
```

- Qwen2.5-7B-Instruct and Qwen2.5-3B-Instruct in 4-bit, greedy decoding, both on one Kaggle T4
  ([notebooks/kaggle_run.ipynb](notebooks/kaggle_run.ipynb): set `STAGE` and `MODEL`, then *Save & Run All*).
- Two prompting methods: LLM-Mob (recent history) and retrieved demonstrations (Jaccard, via Member 2's
  `retrieve_demonstrations`). Neither ever sees the target place.
- E1 heuristic floor, E2 revisit vs new and E3 name test (names present / removed / aliased on Massive-STEPS Tokyo),
  with bootstrap CIs, paired differences, McNemar tests and Holm correction.
- Resumable runs: `results/llm/<dataset>/<run>.jsonl` plus a `.meta.json` with the GPU, versions and a settings fingerprint.
- Results: [reports/llm/results.md](reports/llm/results.md) and `reports/llm/figures/`. Perturbed E3 test sets:
  [data/perturbed/msteps_tokyo](data/perturbed/msteps_tokyo).
- Full details, settings and limits: [docs/llm_pipeline.md](docs/llm_pipeline.md).
- The whole project in plain language (goal, how it was built, findings): [docs/project_overview.md](docs/project_overview.md).
- What each dashboard page shows and how to read it: [docs/dashboard_guide.md](docs/dashboard_guide.md).
