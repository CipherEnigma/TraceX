"""End-to-end preprocessing for one dataset.

Outputs in data/processed/<dataset>/:
    checkins.csv              cleaned check-ins with user_id, poi_id, traj_id, split
    pois.csv                  POI table (category, coordinates, train counts, names if any)
    users.csv                 user_id -> raw user id
    trajectories.jsonl        every trajectory (train/val/test) as a list of check-ins
    instances_val.jsonl       all validation instances
    instances_test.jsonl      all test instances
    sample_test_500.jsonl     fixed-seed evaluation sample (seed 42)
    sample_test_debug20.jsonl 20-instance debug subset of the sample
    stats.json                statistics, pipeline log, POI coverage
    leakage_report.json       results of every leakage / integrity check
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import DATASETS, PROCESSED_DIR, RANDOM_SEED, DatasetConfig
from .instances import build_instances, build_trajectory_records, ids_fingerprint, make_samples
from .leakage import run_checks
from .loaders import load_raw
from .preprocess import assign_split, build_trajectories, clean, frequency_filter, remap_ids
from .stats import compute_stats


def write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process(cfg: DatasetConfig, out_root: Path = PROCESSED_DIR) -> dict:
    if not cfg.raw_path.exists():
        raise FileNotFoundError(f"{cfg.raw_path} missing - run `python -m tracex_data download` first")
    out = out_root / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    log: list = []

    raw, meta = load_raw(cfg)
    df = clean(raw, cfg, log)
    df = frequency_filter(df, cfg, log)
    df, cutoffs = assign_split(df, log)
    df = build_trajectories(df, log)
    df, users, pois = remap_ids(df)

    trajs = build_trajectory_records(df, cfg.has_names)
    instances = {s: build_instances(trajs, cfg.name, s) for s in ["val", "test"]}
    samples = make_samples(instances["test"])

    checks = run_checks(df, trajs, instances, samples)
    failed = [c for c in checks if not c["passed"]]
    report = {"dataset": cfg.name, "all_passed": not failed, "checks": checks}

    stats = compute_stats(cfg, meta, log, df, pois, instances, samples, cutoffs)
    stats["sample_fingerprints"] = {k: ids_fingerprint(v) for k, v in samples.items()}
    stats["random_seed"] = RANDOM_SEED

    cols = ["user_id", "poi_id", "traj_id", "split", "local_time", "category_id", "category", "lat", "lon"]
    cols += [c for c in ["name", "address"] if c in df]
    df[cols].to_csv(out / "checkins.csv", index=False)
    pois.to_csv(out / "pois.csv", index=False)
    users.to_csv(out / "users.csv", index=False)
    write_jsonl(out / "trajectories.jsonl", trajs)
    for s, rows in instances.items():
        write_jsonl(out / f"instances_{s}.jsonl", rows)
    for k, rows in samples.items():
        write_jsonl(out / f"{k}.jsonl", rows)
        (out / f"{k}_ids.txt").write_text("\n".join(r["instance_id"] for r in rows) + "\n", encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if failed:
        raise RuntimeError(f"{cfg.name}: leakage checks failed: {[c['name'] for c in failed]}")
    return stats


def process_many(names) -> dict:
    results = {}
    for n in names:
        print(f"[pipeline] processing {n} ...")
        st = process(DATASETS[n])
        o = st["overall"]
        print(f"[pipeline] {n}: {o['users']} users, {o['pois']} POIs, {o['checkins']} check-ins, "
              f"{o['trajectories']} trajectories, {st['instances']['test']['n']} test instances "
              f"(revisit {st['instances']['test']['revisit_rate']:.1%}) - all leakage checks passed")
        results[n] = st
    return results
