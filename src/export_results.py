"""Export the complete CoLayout benchmark to IEEE tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import evaluate_layout
from src.generate_briefs import DEFAULT_PATH, generate_briefs
from src.methods.blackboard import ARCHITECT, GEOMETRY, STYLE, VISUAL, _initial_geometry, _patched
from src.methods.single_agent_bb_detector import (
    ALL_FIELDS as BB_ALL_FIELDS,
    CONFLICT_TYPES as BB_CONFLICT_TYPES,
    DESIGNER as BB_DESIGNER,
    _initial_all_fields,
)
from src.run_experiments import MEAN_FIELDS, METHODS, METRICS
from src.schema import Conflict, LayoutState

ROOT = Path(__file__).resolve().parents[1]


def _pre_arbiter_csr(row: dict, brief: dict) -> float:
    """Compute exact pre-arbiter CSR from recorded history when available."""
    history = row.get("history")
    method = row.get("method")
    if not isinstance(history, list) or not history:
        return float(row["metrics"]["CSR"])
    if method == "single_agent_detector":
        designer_turns = [h for h in history if h.get("role") == "Layout Designer" and isinstance(h.get("response"), dict)]
        if not designer_turns:
            return float(row["metrics"]["CSR"])
        state = LayoutState.model_validate(designer_turns[-1]["response"])
        return float(evaluate_layout(state)["metrics"]["CSR"])
    if method == "single_agent_bb_detector":
        state = None
        last_hard: list[Conflict] = []
        for entry in history:
            role = entry.get("role")
            rnd = entry.get("round")
            if role == BB_DESIGNER and rnd == 0:
                state = _initial_all_fields(entry["response"], brief)
            elif role == "Compliance Critic" and isinstance(entry.get("conflicts"), list):
                last_hard = [
                    Conflict.model_validate(c) for c in entry["conflicts"] if c.get("severity") == "hard"
                ]
            elif role == BB_DESIGNER and isinstance(rnd, int) and rnd >= 1 and state is not None:
                routed = [c for c in last_hard if c.conflict_type in BB_CONFLICT_TYPES]
                ids = {i for c in routed for i in c.element_ids}
                state = _patched(entry["response"], state, brief, ids, BB_ALL_FIELDS)
            elif role == "Priority Arbiter":
                break
        if state is not None:
            return float(evaluate_layout(state)["metrics"]["CSR"])
    if method in ("ours_full", "ours_no_detector"):
        state = None
        last_hard: list[Conflict] = []
        for entry in history:
            role = entry.get("role")
            rnd = entry.get("round")
            if role == ARCHITECT and rnd == 0:
                state = _initial_geometry(entry["response"], brief)
            elif role == VISUAL and rnd == 0 and state is not None:
                ids = {e.id for e in state.elements}
                state = _patched(entry["response"], state, brief, ids, STYLE)
            elif role == "Compliance Critic":
                if isinstance(entry.get("conflicts"), list):
                    last_hard = [
                        Conflict.model_validate(c) for c in entry["conflicts"] if c.get("severity") == "hard"
                    ]
                elif isinstance(entry.get("response"), dict):
                    last_hard = [
                        Conflict.model_validate({**c, "severity": "hard"})
                        for c in entry["response"].get("conflicts", [])
                    ]
            elif role in (ARCHITECT, VISUAL) and isinstance(rnd, int) and rnd >= 1 and state is not None:
                fields = GEOMETRY if role == ARCHITECT else STYLE
                types = {"overflow", "overlap"} if role == ARCHITECT else {"contrast", "min_font"}
                ids = {i for c in last_hard if c.conflict_type in types for i in c.element_ids}
                state = _patched(entry["response"], state, brief, ids, fields)
            elif role == "Priority Arbiter":
                break
        if state is not None:
            return float(evaluate_layout(state)["metrics"]["CSR"])
    return float(row["metrics"]["CSR"])


def load_results(results_dir: Path) -> tuple[dict, dict, bool]:
    """Read paired checkpoints and check the CSV means against their source rows."""
    briefs = json.loads(DEFAULT_PATH.read_bytes()) if DEFAULT_PATH.is_file() else generate_briefs()
    briefs_by_id = {b["task_id"]: b for b in briefs}
    tasks = {b["task_id"]: b["difficulty"] for b in briefs}
    expected = {(method, task) for method in METHODS for task in tasks}
    records = {}
    samples = defaultdict(list)
    configs = set()
    paths = sorted((results_dir / "raw").glob("*.json"))
    if not paths:
        raise ValueError(f"No raw checkpoints found in {results_dir / 'raw'}.")
    for path in paths:
        try:
            row = json.loads(path.read_bytes())
            method, task, level = row["method"], row["task_id"], row["difficulty"]
            key = (method, task)
            if key not in expected or key in records or level != tasks[task]:
                raise ValueError(f"unknown, duplicate, or mislabeled checkpoint {key}")
            if not isinstance(row["model"], str) or not row["model"].strip():
                raise ValueError("missing model identifier")
            configs.add((row["model"], row["max_rounds"]))
            data = {metric: float(row["metrics"][metric]) for metric in METRICS}
            data.update({name: float(row[name]) for name in ("rounds_used", "total_tokens")})
            if not all(math.isfinite(x) and x >= 0 for x in data.values()):
                raise ValueError("metrics and costs must be finite and nonnegative")
            if any(data[name] > 1 for name in ("BOR", "CPR", "AS")) or data["CSR"] not in (0, 1):
                raise ValueError("invalid per-task metric range")
            data["pre_CSR"] = _pre_arbiter_csr(row, briefs_by_id[task])
            samples[method, level].append(data)
            # Long conversation histories are unnecessary for figure generation.
            records[key] = {k: row[k] for k in (
                "method", "task_id", "difficulty", "metrics", "final_state", "conflicts",
            )}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid checkpoint {path.name}: {exc}") from exc
    missing = expected - records.keys()
    if missing:
        method, task = min(missing)
        raise ValueError(f"Incomplete benchmark: missing {len(missing)} checkpoints, including {method}/{task}.")
    if len(configs) != 1:
        raise ValueError("Checkpoints mix models or round budgets. Export one benchmark run at a time.")
    model, _ = configs.pop()
    manifest = results_dir / "PROVENANCE.json"
    if manifest.is_file() and json.loads(manifest.read_bytes()).get("model") != model:
        raise ValueError("PROVENANCE.json and the checkpoints identify different models.")
    model_tags = model.casefold().replace("_", "-").split("-")
    mock = "mock" in model_tags or "synthetic" in model_tags

    groups = {}
    with (results_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), start=2):
            try:
                key = (row["method"], row["difficulty"])
                if key in groups or key not in samples:
                    raise ValueError(f"duplicate or unknown group {key}")
                count = int(row["n_tasks"])
                if count != len(samples[key]):
                    raise ValueError(f"wrong task count for {key}")
                values = {field: float(row[field]) for field in MEAN_FIELDS}
                for field, value in values.items():
                    mean = fmean(s[field] for s in samples[key])
                    if not math.isclose(value, mean, rel_tol=1e-9, abs_tol=1e-12):
                        raise ValueError(f"{key} {field} differs from the raw checkpoints")
                values["pre_CSR"] = fmean(s["pre_CSR"] for s in samples[key])
                groups[key] = dict(n_tasks=count, **values)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid summary.csv line {line}: {exc}") from exc
    if groups.keys() != samples.keys():
        raise ValueError("Incomplete summary.csv. Rebuild it from the completed checkpoints.")
    return records, groups, mock


def export_results(results_dir: Path, paper_dir: Path, case_task_id: str | None = None) -> str:
    """Build all six files before replacing any existing paper artifacts."""
    records, groups, mock = load_results(results_dir)
    from src.paper_case import export_case
    from src.paper_quantitative import plot_performance, write_tables

    paper_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=paper_dir) as temp:
        stage = Path(temp)
        figures, tables = stage / "figures", stage / "tables"
        figures.mkdir()
        tables.mkdir()
        selected = export_case(records, case_task_id, figures, mock=mock)
        plot_performance(groups, figures, mock=mock)
        write_tables(groups, tables, mock=mock)
        for folder in (tables, figures):
            destination = paper_dir / folder.name
            destination.mkdir(exist_ok=True)
            for file in folder.iterdir():
                file.replace(destination / file.name)
    print(f"Exported {len(records)} checkpoints to {paper_dir}. Case study: {selected}.")
    if mock:
        print("Offline mock data: artifacts are labeled for pipeline verification.")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper")
    parser.add_argument("--case-task-id", help="Use this task ID for the three-panel case study.")
    args = parser.parse_args()
    try:
        export_results(args.results_dir, args.paper_dir, args.case_task_id)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")


if __name__ == "__main__":
    main()
