import csv
import json

import pytest

from src.export_results import load_results
from src.generate_briefs import generate_briefs
from src.run_experiments import MEAN_FIELDS, METHODS, METRICS


@pytest.fixture
def complete_run(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    summary = []
    for method in METHODS:
        for task in generate_briefs():
            row = {
                "method": method, "task_id": task["task_id"], "difficulty": task["difficulty"],
                "metrics": dict.fromkeys(METRICS, 1), "rounds_used": 2, "total_tokens": 500,
                "conflicts": [], "final_state": {}, "model": "offline-mock", "max_rounds": 5,
            }
            (raw / f"{method}__{task['task_id']}.json").write_text(json.dumps(row), encoding="utf8")
        for level in ("easy", "medium", "hard"):
            summary.append(dict(method=method, difficulty=level, n_tasks=12,
                                **row["metrics"], rounds_used=2, total_tokens=500))
    with (tmp_path / "summary.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("method", "difficulty", "n_tasks", *MEAN_FIELDS))
        writer.writeheader()
        writer.writerows(summary)
    return tmp_path


def test_export_requires_complete_paired_tasks(complete_run):
    records, groups, mock = load_results(complete_run)
    assert (len(records), len(groups), mock) == (len(METHODS) * 36, len(METHODS) * 3, True)
    next((complete_run / "raw").glob("*.json")).unlink()
    with pytest.raises(ValueError, match="Incomplete benchmark"):
        load_results(complete_run)


def test_export_rejects_stale_summary(complete_run):
    path = complete_run / "summary.csv"
    path.write_text(path.read_text().replace(",500", ",499", 1), encoding="utf8")
    with pytest.raises(ValueError, match="total_tokens differs"):
        load_results(complete_run)


def test_export_rejects_mixed_model_provenance(complete_run):
    path = next((complete_run / "raw").glob("*.json"))
    record = json.loads(path.read_bytes())
    record["model"] = "different-model"
    path.write_text(json.dumps(record), encoding="utf8")
    with pytest.raises(ValueError, match="mix models"):
        load_results(complete_run)
