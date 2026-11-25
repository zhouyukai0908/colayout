from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.llm_client import InvalidCompletion, LLMClient, model_name


class Answer(BaseModel):
    width: int


def scripted_client(*outputs):
    responses = iter(outputs)
    requests = []

    def create(**request):
        requests.append(request)
        content = next(responses)
        if isinstance(content, Exception):
            raise content
        if isinstance(content, SimpleNamespace):
            return content
        reply = SimpleNamespace(message=SimpleNamespace(content=content))
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10)
        return SimpleNamespace(choices=[reply], usage=usage)

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return LLMClient(client=sdk), requests


def test_retries_json_and_schema_failures_counting_each_attempt(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "local-model")
    llm, requests = scripted_client("{", '{"width": "invalid"}', '{"width": 12}')
    original = [{"role": "user", "content": "Choose a width."}]
    data, usage = llm.complete_json(original, validator=Answer)
    assert data == {"width": 12}
    assert usage == {"prompt_tokens": 21, "completion_tokens": 9, "total_tokens": 30}
    assert len(original) == 1 and len(requests) == 3
    assert "Correct the validation error" in requests[-1]["messages"][-1]["content"]
    assert all(r["response_format"] == {"type": "json_object"} for r in requests)
    assert all(r["model"] == "local-model" for r in requests)


def test_bad_top_level_value_stops_after_two_retries():
    llm, requests = scripted_client("null", "[]", "42")
    with pytest.raises(InvalidCompletion) as exc:
        llm.complete_json([{"role": "user", "content": "Return JSON."}])
    assert len(requests) == 3 and exc.value.token_usage["total_tokens"] == 30


def test_clean_response_uses_one_call_and_default_model(monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    llm, requests = scripted_client('{"valid": true}')
    assert model_name() == "gpt-4o-mini"
    data, usage = llm.complete_json([{"role": "user", "content": "Return JSON."}])
    assert data == {"valid": True} and usage["total_tokens"] == 10
    assert len(requests) == 1
    assert "max_tokens" not in requests[0]


def test_cumulative_usage_across_success_retries_and_refusal():
    refusal = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(refusal="Declined"))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=2, total_tokens=13),
    )
    llm, requests = scripted_client("{}", "{", "[]", "42", refusal)
    _, first = llm.complete_json([])
    snapshot = llm.cumulative_usage
    with pytest.raises(InvalidCompletion) as exc:
        llm.complete_json([], max_tokens=64)
    assert exc.value.token_usage["total_tokens"] == 30
    assert all(request["max_tokens"] == 64 for request in requests[1:])
    with pytest.raises(RuntimeError, match="declined"):
        llm.complete_json([])
    assert snapshot == first == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert llm.cumulative_usage == {"prompt_tokens": 39, "completion_tokens": 14, "total_tokens": 53}
    assert llm.unreported_requests == 0


def test_unknown_usage_counts_attempts_and_preserves_reported_fields():
    partial = SimpleNamespace(usage=SimpleNamespace(completion_tokens=2, total_tokens=8))
    llm, requests = scripted_client("{", TimeoutError("timeout"), partial, SimpleNamespace())
    with pytest.raises(TimeoutError):
        llm.complete_json([])
    with pytest.raises(RuntimeError, match="prompt_tokens"):
        llm.complete_json([])
    with pytest.raises(RuntimeError, match="total_tokens"):
        llm.complete_json([])
    assert len(requests) == 4
    assert llm.cumulative_usage == {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 18}
    assert llm.unreported_requests == 3


@pytest.mark.parametrize("limit", [False, -1, 0, 1.5])
def test_invalid_completion_limit_fails_before_request(limit):
    llm, requests = scripted_client()
    with pytest.raises(ValueError, match="max_tokens"):
        llm.complete_json([], max_tokens=limit)
    assert requests == []
    assert llm.unreported_requests == 0
