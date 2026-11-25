"""Run benchmark methods with one durable JSON checkpoint per method and task."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile
from threading import Event, local
from uuid import uuid4

# Support both python -m src.run_experiments and python src/run_experiments.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate_briefs import DEFAULT_PATH, generate_briefs
from src.llm_client import InvalidCompletion, LLMClient, model_name
from src.methods import (
    naive_multi_agent, ours_full, ours_no_detector, self_refine, single_agent, single_agent_detector,
)


METHODS = {
    "single_agent": single_agent.run_method,
    "self_refine": self_refine.run_method,
    "naive_multi_agent": naive_multi_agent.run_method,
    "single_agent_detector": single_agent_detector.run_method,
    "ours_no_detector": ours_no_detector.run_method,
    "ours_full": ours_full.run_method,
}
METRICS = ("OR", "BOR", "CPR", "AS", "CSR")
MEAN_FIELDS = (*METRICS, "rounds_used", "total_tokens")


def _methods(value: str) -> list[str]:
    if value.strip() == "all":
        return list(METHODS)
    names = list(dict.fromkeys(part.strip() for part in value.split(",")))
    unknown = set(names) - METHODS.keys()
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown methods {sorted(unknown)}. Choose from {list(METHODS)}.")
    return names


def _atomic_write(path: Path, content: str) -> None:
    # A stopped process can leave a .tmp file, which is never treated as a result.
    tmp = NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".tmp",
                             prefix=f".{path.stem}-", dir=path.parent, delete=False)
    tmp_path = Path(tmp.name)
    try:
        with tmp:
            tmp.write(content)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _summary_row(record: dict) -> dict:
    return {
        "method": record["method"], "difficulty": record["difficulty"],
        **{name: record["metrics"][name] for name in METRICS},
        "rounds_used": record["rounds_used"], "total_tokens": record["total_tokens"],
    }


def _save_summary(rows, path: Path) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row["difficulty"]].append(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("method", "difficulty", "n_tasks", *MEAN_FIELDS))
    writer.writeheader()
    for (method, level), group in sorted(groups.items()):
        means = {name: fmean(row[name] for row in group) for name in MEAN_FIELDS}
        writer.writerow({"method": method, "difficulty": level, "n_tasks": len(group), **means})
    _atomic_write(path, stream.getvalue())


def _load_cache(raw_dir: Path, config: dict, briefs_by_id: dict) -> dict:
    rows = {}
    for path in sorted(raw_dir.glob("*.json")):
        try:
            record = json.loads(path.read_bytes())
            if any(record[key] != value for key, value in config.items()):
                raise ValueError("model or max_rounds differs; choose a separate --out-dir")
            method, task_id = record["method"], record["task_id"]
            brief = briefs_by_id[task_id]
            if (method not in METHODS or path.name != f"{method}__{task_id}.json"
                    or record["difficulty"] != brief["difficulty"] or record["domain"] != brief["domain"]):
                raise ValueError("checkpoint identity differs from the benchmark")
            for field in ("final_state", "conflicts", "history"):
                if field not in record:
                    raise ValueError(f"missing {field}")
            rows[method, task_id] = _summary_row(record)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid checkpoint {path}: {exc}") from exc
    return rows


def _run_one(method: str, brief: dict, config: dict, raw_dir: Path, clients) -> dict:
    # Each worker reuses its own client; its usage counters never span concurrent jobs.
    if not hasattr(clients, "llm"):
        clients.llm = LLMClient()
    llm = clients.llm
    task_id = brief["task_id"]
    print(f"Running {method} {task_id}", flush=True)
    metadata = {
        **config, "method": method, "task_id": task_id,
        "difficulty": brief["difficulty"], "domain": brief["domain"],
    }
    before = dict(getattr(llm, "cumulative_usage", {}))
    unknown_before = getattr(llm, "unreported_requests", 0)
    try:
        result = METHODS[method](brief, llm, max_rounds=config["max_rounds"])
    except (Exception, KeyboardInterrupt) as exc:
        usage = {k: llm.cumulative_usage[k] - v for k, v in before.items()}
        unknown = getattr(llm, "unreported_requests", 0) - unknown_before
        failure = {
            **metadata, "status": "failed", "error_type": type(exc).__name__,
            "round_index": getattr(exc, "colayout_round", None),
            "role": getattr(exc, "colayout_role", None),
            "token_usage": usage or None, "total_tokens": usage.get("total_tokens"),
            "unreported_requests": unknown, "usage_complete": bool(usage) and unknown == 0,
        }
        if isinstance(exc, InvalidCompletion):
            failure["reason"] = str(exc)
        failure_dir = raw_dir.parent / "failures"
        failure_dir.mkdir(exist_ok=True)
        failure_path = failure_dir / f"{method}__{task_id}__{uuid4().hex}.json"
        _atomic_write(failure_path, json.dumps(failure, indent=2, ensure_ascii=False) + "\n")
        print(f"Failed attempt saved to {failure_path}", file=sys.stderr, flush=True)
        raise
    record = {**result, **metadata}
    if before:
        record["token_usage"] = {k: llm.cumulative_usage[k] - v for k, v in before.items()}
    row = _summary_row(record)
    path = raw_dir / f"{method}__{task_id}.json"
    _atomic_write(path, json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return row


def run_experiments(methods: list[str], limit: int | None, max_rounds: int, out_dir: Path,
                    *, workers: int = 1, continue_on_invalid: bool = False) -> Path:
    """Save selected runs and aggregate every completed checkpoint in the directory."""
    if max_rounds < 1 or (limit is not None and limit < 0):
        raise ValueError("max_rounds must be positive and limit must be nonnegative")
    if not methods or any(name not in METHODS for name in methods):
        raise ValueError(f"methods must be selected from {list(METHODS)}")
    if workers < 1:
        raise ValueError("workers must be positive")
    briefs = json.loads(DEFAULT_PATH.read_bytes()) if DEFAULT_PATH.is_file() else generate_briefs()
    briefs_by_id = {b["task_id"]: b for b in briefs}
    if len(briefs_by_id) != len(briefs) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", task_id) for task_id in briefs_by_id
    ):
        raise ValueError("Benchmark task IDs must be unique and filename-safe")

    config = {"model": model_name(), "max_rounds": max_rounds}
    raw_dir = Path(out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_path = raw_dir.parent / "summary.csv"
    rows = _load_cache(raw_dir, config, briefs_by_id)
    _save_summary(rows.values(), summary_path)
    jobs = []
    for method in dict.fromkeys(methods):
        for brief in briefs[:limit]:
            task_id = brief["task_id"]
            if (method, task_id) in rows:
                print(f"Cached {method} {task_id}", flush=True)
                continue
            jobs.append((method, brief))

    clients = local()
    invalid = []
    stop = Event()

    def attempt(job):
        method, brief = job
        try:
            return _run_one(method, brief, config, raw_dir, clients)
        except InvalidCompletion:
            if not continue_on_invalid:
                stop.set()
                raise
            return None
        except (Exception, KeyboardInterrupt):
            stop.set()
            raise

    def record(job, row):
        key = job[0], job[1]["task_id"]
        if row is None:
            invalid.append("/".join(key))
        else:
            rows[key] = row
            _save_summary(rows.values(), summary_path)

    if workers == 1:
        for job in jobs:
            record(job, attempt(job))
    else:
        # Submit at most one outstanding job per worker. On fatal errors, drain
        # already dispatched work and persist its results without dispatching more.
        work = iter(jobs)
        fatal = None
        active = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                while True:
                    while not stop.is_set() and len(active) < workers:
                        job = next(work, None)
                        if job is None:
                            break
                        active[pool.submit(attempt, job)] = job
                    if not active:
                        break
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        job = active.pop(future)
                        try:
                            row = future.result()
                        except (Exception, KeyboardInterrupt) as exc:
                            if fatal is None:
                                fatal = exc
                        else:
                            record(job, row)
            except KeyboardInterrupt:
                # Workers may finish after Ctrl-C. Recover their durable checkpoints.
                pool.shutdown(wait=True, cancel_futures=True)
                _save_summary(_load_cache(raw_dir, config, briefs_by_id).values(), summary_path)
                raise
        if fatal is not None:
            raise fatal
    if invalid:
        raise ValueError(f"Incomplete benchmark: {len(invalid)} invalid run(s): {', '.join(invalid)}")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=_methods, default="all", metavar="NAMES",
                        help="Comma-separated method names or all (default).")
    parser.add_argument("--limit", type=int, help="First N tasks per selected method.")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--workers", type=int, default=1, help="Concurrent independent method/task pairs.")
    parser.add_argument("--continue-on-invalid", action="store_true",
                        help="Finish other selected pairs after JSON validation retries are exhausted.")
    args = parser.parse_args()
    try:
        summary = run_experiments(args.methods, args.limit, args.max_rounds, args.out_dir,
                                  workers=args.workers, continue_on_invalid=args.continue_on_invalid)
    except ValueError as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
