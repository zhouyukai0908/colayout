import csv
import json
import subprocess
import sys
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from src import run_experiments as runner
from src.llm_client import InvalidCompletion, LLMClient


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_resumes_after_interruption_and_summarizes_all_saved_methods(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "offline-model")
    monkeypatch.setattr(runner, "LLMClient", object)
    called = []

    def run(brief, llm, max_rounds):
        called.append(brief["task_id"])
        if len(called) == 2:
            raise KeyboardInterrupt()
        index = int(brief["task_id"][-2:])
        return {
            "final_state": {}, "history": [], "conflicts": [],
            "rounds_used": index, "total_tokens": index * 10,
            "metrics": {"OR": index * 0.1, "BOR": 0, "CPR": 1, "AS": 1, "CSR": 0},
        }

    for name in ("single_agent", "self_refine"):
        monkeypatch.setitem(runner.METHODS, name, run)

    with pytest.raises(KeyboardInterrupt):
        runner.run_experiments(["single_agent"], 2, 5, tmp_path)
    first, = (tmp_path / "raw").glob("*.json")
    original = first.read_bytes()
    assert read_csv(tmp_path / "summary.csv")[0]["n_tasks"] == "1"
    (tmp_path / "raw" / ".partial.tmp").write_text("{", encoding="utf-8")
    runner.run_experiments(["single_agent"], 2, 5, tmp_path)
    assert called == ["poster_easy_01", "poster_easy_02", "poster_easy_02"]
    assert first.read_bytes() == original
    runner.run_experiments(["self_refine"], 1, 5, tmp_path)
    rows = {r["method"]: r for r in read_csv(tmp_path / "summary.csv")}
    assert rows["self_refine"]["n_tasks"] == "1"
    assert rows["single_agent"]["n_tasks"] == "2"
    assert float(rows["single_agent"]["OR"]) == pytest.approx(0.15)
    assert float(rows["single_agent"]["rounds_used"]) == 1.5
    assert float(rows["single_agent"]["total_tokens"]) == 15
    assert len(list((tmp_path / "raw").glob("*.json"))) == 3
    assert json.loads(original)["model"] == "offline-model"

    def require_offline():
        pytest.fail("Cached runs should finish without constructing a client")

    monkeypatch.setattr(runner, "LLMClient", require_offline)
    (tmp_path / "summary.csv").unlink()
    summary = runner.run_experiments(["single_agent"], 1, 5, tmp_path)
    assert {r["method"]: r for r in read_csv(summary)} == rows
    with pytest.raises(ValueError, match="max_rounds"):
        runner.run_experiments(["self_refine"], 1, 4, tmp_path)
    monkeypatch.setenv("MODEL_NAME", "different-model")
    with pytest.raises(ValueError, match="model"):
        runner.run_experiments(["self_refine"], 1, 5, tmp_path)


def test_raw_directory_with_corrupt_checkpoint_is_reported(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "single_agent__poster_easy_01.json").write_bytes(b"{")
    with pytest.raises(ValueError, match="Invalid checkpoint.*poster_easy_01"):
        runner.run_experiments(["single_agent"], 1, 5, tmp_path)


def test_cli_starts_from_an_unrelated_working_directory(tmp_path):
    script = Path(runner.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--limit", "0", "--out-dir", "demo"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "summary.csv" in result.stdout
    assert read_csv(tmp_path / "demo" / "summary.csv") == []


def test_failures_record_prior_roles_and_all_rejected_retries(tmp_path, monkeypatch):
    # One completed method, then one valid role call and three invalid replies per failed attempt.
    replies = iter(['{}', '{}', 'null', '[]', '{', '{}', 'null', '[]', '{'])

    def create(**request):
        message = SimpleNamespace(content=next(replies))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(
            prompt_tokens=6, completion_tokens=2, total_tokens=8,
        ))

    client = LLMClient(client=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(runner, "LLMClient", lambda: client)
    messages = [{"role": "user", "content": "Return JSON."}]

    def valid(brief, llm, max_rounds):
        _, usage = llm.complete_json(messages)
        return dict(final_state={}, metrics=dict.fromkeys(runner.METRICS, 0), conflicts=[],
                    rounds_used=1, total_tokens=usage["total_tokens"], history=[])

    def invalid(brief, llm, max_rounds):
        llm.complete_json(messages)
        llm.complete_json(messages)

    monkeypatch.setitem(runner.METHODS, "single_agent", valid)
    monkeypatch.setitem(runner.METHODS, "self_refine", invalid)
    for _ in range(2):
        with pytest.raises(InvalidCompletion):
            runner.run_experiments(["single_agent", "self_refine"], 1, 5, tmp_path)
    failures = [json.loads(p.read_bytes()) for p in (tmp_path / "failures").glob("*.json")]
    assert len(failures) == 2
    for failure in failures:
        assert failure["method"] == "self_refine" and failure["usage_complete"]
        assert failure["token_usage"] == {"prompt_tokens": 24, "completion_tokens": 8, "total_tokens": 32}
        assert failure["total_tokens"] == 32 and failure["error_type"] == "InvalidCompletion"
    completed, = (tmp_path / "raw").glob("*.json")
    assert json.loads(completed.read_bytes())["token_usage"]["total_tokens"] == 8
    assert len(read_csv(tmp_path / "summary.csv")) == 1


def _offline_result(tokens):
    return dict(final_state={}, metrics=dict.fromkeys(runner.METRICS, 0), conflicts=[],
                rounds_used=1, total_tokens=tokens, history=[])


@pytest.mark.parametrize("workers", [1, 2])
def test_continues_after_invalid_and_preserves_per_task_usage(tmp_path, monkeypatch, workers):
    class Client:
        def __init__(self):
            self.cumulative_usage = {"total_tokens": 0}

    first_pair = Barrier(workers)
    clients = {}

    def run(brief, llm, max_rounds):
        index = int(brief["task_id"][-2:])
        clients[index] = llm
        llm.cumulative_usage["total_tokens"] += index
        if index <= workers:
            first_pair.wait(timeout=5)
        if index == 1:
            raise InvalidCompletion("malformed document", {"total_tokens": index})
        return _offline_result(index)

    monkeypatch.setattr(runner, "LLMClient", Client)
    monkeypatch.setitem(runner.METHODS, "single_agent", run)
    with pytest.raises(ValueError, match="1 invalid run.*single_agent/poster_easy_01"):
        runner.run_experiments(["single_agent"], 3, 5, tmp_path,
                               workers=workers, continue_on_invalid=True)
    success = [json.loads(p.read_bytes()) for p in sorted((tmp_path / "raw").glob("*.json"))]
    assert [s["token_usage"]["total_tokens"] for s in success] == [2, 3]
    assert len(set(clients.values())) == workers
    failed, = (tmp_path / "failures").glob("*.json")
    assert json.loads(failed.read_bytes())["token_usage"]["total_tokens"] == 1
    assert read_csv(tmp_path / "summary.csv")[0]["n_tasks"] == "2"


def test_fatal_error_stops_dispatch_and_drains_active_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "LLMClient", object)
    started = Barrier(2)
    drained = Event()
    called = []
    original_wait = runner.wait

    def wait_until_draining(futures, **kwargs):
        if len(futures) == 1:
            drained.set()
        return original_wait(futures, **kwargs)

    def run(brief, llm, max_rounds):
        index = int(brief["task_id"][-2:])
        called.append(index)
        started.wait(timeout=5)
        if index == 1:
            raise RuntimeError("endpoint unavailable")
        assert drained.wait(5)
        return _offline_result(4)

    monkeypatch.setattr(runner, "wait", wait_until_draining)
    monkeypatch.setitem(runner.METHODS, "single_agent", run)
    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        runner.run_experiments(["single_agent"], 4, 5, tmp_path,
                               workers=2, continue_on_invalid=True)
    assert sorted(called) == [1, 2]
    assert [p.name for p in (tmp_path / "raw").glob("*.json")] == ["single_agent__poster_easy_02.json"]
    assert read_csv(tmp_path / "summary.csv")[0]["n_tasks"] == "1"
