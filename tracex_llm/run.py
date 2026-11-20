"""Resumable LLM prediction runner (Member 3).

    # always debug on 20 instances first (brief)
    python -m tracex_llm.run --dataset nyc --model qwen7b --method llmmob --sample sample_test_debug20
    # several runs share one loaded model
    python -m tracex_llm.run --dataset nyc tky --model qwen7b --method llmmob retrieval
    # E3 on Massive-STEPS Tokyo, all name conditions
    python -m tracex_llm.run --dataset msteps_tokyo --model qwen7b --method llmmob retrieval --condition all
    # see the first prompt without loading a model
    python -m tracex_llm.run --dataset msteps_tokyo --method retrieval --condition present --dry-run

Each run writes `results/llm/<dataset>/<run>.jsonl` (one record per instance, appended and flushed
after every batch) and `<run>.meta.json` (backend info + settings fingerprint). Re-running the same
command resumes where it stopped; changing a setting mid-run is refused unless --overwrite is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

from tracex_data.api import load_dataset
from tracex_ir import metrics as M
from tracex_ir.retrieval import IDF

from . import config as C
from .backends import load_backend
from .parse import parse_response, user_poi_ids
from .prompts import SYSTEM_PROMPT, build_prompt, present_names


# ---------------------------------------------------------------------------
# Settings fingerprint and name conditions
# ---------------------------------------------------------------------------
def settings(model: str, method: str, retriever: str | None, condition: str) -> dict:
    """Everything that changes a prediction. Stored with the run and checked on resume."""
    return {
        "model": model, "hf_id": C.MODELS[model].hf_id, "load_in_4bit": C.MODELS[model].load_in_4bit,
        "quantization": C.QUANTIZATION, "generation": C.GENERATION, "batch_size": C.BATCH_SIZE,
        "method": method, "retriever": retriever if method == "retrieval" else None,
        "demo_top_k": C.DEMO_TOP_K if method == "retrieval" else None,
        "condition": condition, "n_predictions": C.N_PREDICTIONS,
        "llmmob_max_history": C.LLMMOB_MAX_HISTORY, "demo_max_context": C.DEMO_MAX_CONTEXT,
        "prefix_max_checkins": C.PREFIX_MAX_CHECKINS, "include_target_time": C.INCLUDE_TARGET_TIME,
        "system_prompt": SYSTEM_PROMPT,
    }


def fingerprint(s: dict) -> str:
    return hashlib.sha1(json.dumps(s, sort_keys=True).encode()).hexdigest()[:12]


def name_fn_for(dataset: str, condition: str):
    if condition in (C.NO_NAMES, "removed"):
        return None
    if condition == "present":
        return present_names
    if condition == "aliased":
        raise NotImplementedError("the 'aliased' condition is added with tracex_llm.aliasing (E3)")
    raise ValueError(f"unknown condition {condition!r}")


# ---------------------------------------------------------------------------
# Resumable JSONL
# ---------------------------------------------------------------------------
def load_done(path) -> dict[str, dict]:
    """Records already written. A truncated last line (crash mid-write) is removed from the file."""
    if not path.exists():
        return {}
    good, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError:
                dropped += 1
    if dropped:
        print(f"  [resume] dropping {dropped} corrupt line(s) from {path.name}")
        with open(path, "w", encoding="utf-8") as f:
            for r in good:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {r["instance_id"]: r for r in good}


def append(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
class Session:
    """Holds one loaded backend and the per-dataset objects shared across runs."""

    def __init__(self, model: str, batch_size: int | None = None):
        self.model = model
        self.batch_size = batch_size or C.BATCH_SIZE
        self._llm = None
        self._data = {}

    @property
    def llm(self):
        if self._llm is None:  # load lazily: skipped entirely if every run is already complete
            print(f"[model] loading {self.model} ...")
            t0 = time.perf_counter()
            kw = {} if C.MODELS[self.model].backend == "mock" else {"batch_size": self.batch_size}
            self._llm = load_backend(self.model, **kw)
            print(f"[model] loaded in {time.perf_counter() - t0:.0f}s: {self._llm.info()}")
        return self._llm

    def data(self, dataset):
        if dataset not in self._data:
            ds = load_dataset(dataset)
            self._data[dataset] = (ds, IDF(ds.train_trajectories()), set(ds.pois["poi_id"].astype(int)))
        return self._data[dataset]


def run_one(sess: Session, dataset: str, method: str, condition: str, retriever: str | None,
            sample: str, limit: int | None = None, overwrite: bool = False, dry_run: bool = False):
    retriever = (retriever or C.DEFAULT_RETRIEVER) if method == "retrieval" else None
    name = C.run_name(dataset, sess.model, method, condition, retriever, sample)
    path = C.result_path(dataset, sess.model, method, condition, retriever, sample)
    meta_path = path.with_suffix(".meta.json")
    ds, idf, valid = sess.data(dataset)
    name_fn = name_fn_for(dataset, condition)
    insts = ds.instances(sample)[:limit] if limit else ds.instances(sample)

    if dry_run:
        inst = insts[0]
        p = build_prompt(method, inst, ds.history(inst), idf, retriever or C.DEFAULT_RETRIEVER, name_fn)
        print(f"=== {name} (dry run, {len(insts)} instances)\n--- system\n{p.system}\n--- user\n{p.user}\n"
              f"--- meta\n{json.dumps(p.meta, ensure_ascii=False)}")
        return None

    s = settings(sess.model, method, retriever, condition)
    fp = fingerprint(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for f in (path, meta_path):
            f.unlink(missing_ok=True)
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fp:
            raise SystemExit(f"[{name}] settings changed since this run started "
                             f"({old['fingerprint']} -> {fp}). Use --overwrite to restart it.")
    done = load_done(path)
    todo = [i for i in insts if i["instance_id"] not in done]
    print(f"=== {name}: {len(done)} done, {len(todo)} to go")
    if not todo:
        return summarize(name, path)

    llm = sess.llm
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
        "run": name, "fingerprint": fp, "settings": s, "sessions": []}
    meta["sessions"].append({"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "backend": llm.info(), "todo": len(todo)})
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    t_start, n_done = time.perf_counter(), 0
    bs = sess.batch_size
    for b in range(0, len(todo), bs):
        batch = todo[b:b + bs]
        hists = [ds.history(i) for i in batch]
        prompts = [build_prompt(method, i, h, idf, retriever or C.DEFAULT_RETRIEVER, name_fn)
                   for i, h in zip(batch, hists)]
        outs = llm.generate([p.messages() for p in prompts])
        recs = []
        for inst, h, p, o in zip(batch, hists, prompts, outs):
            pr = parse_response(o.text, valid, user_poi_ids(inst, h), set(p.meta["prompt_poi_ids"]))
            tgt = inst["target"]["poi_id"]
            recs.append({
                "instance_id": inst["instance_id"], "run": name, "fingerprint": fp,
                "dataset": dataset, "model": sess.model, "method": method, "retriever": retriever,
                "condition": condition, "user_id": inst["user_id"],
                # scoring-only fields (read after the prompt is built, never shown to the model)
                "target": tgt, "is_revisit": inst["is_revisit"],
                "target_in_history": inst["target_in_history"], "target_in_prefix": inst["target_in_prefix"],
                "target_in_prompt": tgt in set(p.meta["prompt_poi_ids"]),
                "raw": o.text, "parse": pr.to_dict(),
                "prompt_sha1": hashlib.sha1(p.user.encode()).hexdigest()[:12],
                "prompt_meta": {k: v for k, v in p.meta.items() if k != "prompt_poi_ids"},
                "n_prompt_pois": len(p.meta["prompt_poi_ids"]),
                "prompt_tokens": o.prompt_tokens, "new_tokens": o.new_tokens, "seconds": o.seconds,
            })
        append(path, recs)
        n_done += len(batch)
        el = time.perf_counter() - t_start
        rate = el / n_done
        print(f"  [{len(done) + n_done}/{len(insts)}] {rate:.2f}s/inst, "
              f"ETA {rate * (len(todo) - n_done) / 60:.1f} min")

    meta["sessions"][-1].update({"finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "seconds": round(time.perf_counter() - t_start, 1),
                                 "sec_per_instance": round((time.perf_counter() - t_start) / len(todo), 3)})
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return summarize(name, path)


def summarize(name, path) -> dict:
    """Quick look at a finished run. Full evaluation with CIs is in tracex_llm.evaluate."""
    recs = list(load_done(path).values())
    n = len(recs)
    if not n:
        return {}
    r = {
        "n": n,
        "acc1": sum(M.accuracy_at_k(x["parse"]["ranked"], x["target"], 1) for x in recs) / n,
        "acc5": sum(M.accuracy_at_k(x["parse"]["ranked"], x["target"], 5) for x in recs) / n,
        "invalid_rate": sum(x["parse"]["invalid_answer"] for x in recs) / n,
        "copy_rate": sum(bool(x["parse"]["copied_top1"]) for x in recs) / n,
    }
    print(f"  -> {name}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in r.items()))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tracex_llm.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", required=True, choices=["nyc", "tky", "msteps_tokyo"])
    ap.add_argument("--model", default="qwen7b", choices=list(C.MODELS))
    ap.add_argument("--method", nargs="+", default=list(C.METHODS), choices=list(C.METHODS))
    ap.add_argument("--retriever", default=C.DEFAULT_RETRIEVER, choices=list(C.RETRIEVERS))
    ap.add_argument("--condition", nargs="+", default=["default"],
                    help="E3 name conditions for Massive-STEPS: present removed aliased, or 'all'. "
                         "'default' = 'none' for NYC/TKY and 'present' for Massive-STEPS.")
    ap.add_argument("--sample", default="sample_test_500",
                    choices=["sample_test_500", "sample_test_debug20", "val", "test"])
    ap.add_argument("--limit", type=int, help="only the first N instances (debugging)")
    ap.add_argument("--batch-size", type=int, help=f"default {C.BATCH_SIZE}; keep it fixed across reported runs")
    ap.add_argument("--overwrite", action="store_true", help="delete existing results for these runs first")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit (no model)")
    a = ap.parse_args(argv)

    sess = Session(a.model, a.batch_size)
    summary = {}
    for d in a.dataset:
        allowed = C.name_conditions(d)
        if a.condition == ["default"]:
            conds = ["present"] if "present" in allowed else [C.NO_NAMES]
        elif a.condition == ["all"]:
            conds = list(allowed)
        else:
            conds = [c for c in a.condition if c in allowed]
            skipped = [c for c in a.condition if c not in allowed]
            if skipped:
                print(f"[{d}] skipping conditions {skipped}: {d} allows {allowed}")
        for m in a.method:
            for c in conds:
                res = run_one(sess, d, m, c, a.retriever, a.sample, a.limit, a.overwrite, a.dry_run)
                if res:
                    summary[C.run_name(d, a.model, m, c, a.retriever if m == "retrieval" else None,
                                       a.sample)] = res
    if summary:
        print("\n== summary")
        for k, v in summary.items():
            print(f"  {k}: acc1={v['acc1']:.3f} acc5={v['acc5']:.3f} "
                  f"invalid={v['invalid_rate']:.3f} copy={v['copy_rate']:.3f} (n={v['n']})")
    return summary


if __name__ == "__main__":
    main()
