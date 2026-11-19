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
