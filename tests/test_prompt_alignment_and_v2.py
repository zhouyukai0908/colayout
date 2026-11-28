"""Tests for shared rule alignment, zero-silent-drop parsing, sa_twostage_bb_det, and v2 metadata."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from src import run_experiments as runner
from src.export_results import _pre_arbiter_csr
from src.llm_client import InvalidCompletion, LLMClient
from src.methods import (
    ours_full,
    sa_twostage_bb_det,
    single_agent_bb_detector,
    single_agent_detector,
)
from src.methods.rules import (
    GEOMETRY_RULES,
    OUTPUT_FORMAT_BB_SINGLE_INIT_TEMPLATE,
    OUTPUT_FORMAT_GEOMETRY_INIT,
    OUTPUT_FORMAT_PATCH_TEMPLATE,
    REVISION_RULES,
    STYLE_RULES,
)

BRIEF = {
    "task_id": "align_test",
    "domain": "poster",
    "difficulty": "easy",
    "canvas_width": 200,
    "canvas_height": 200,
    "canvas_bg_color": "#FFFFFF",
    "required_elements": [
        {"id": "panel", "type": "card", "text_or_desc": "Dark backdrop"},
        {"id": "headline", "type": "title", "text_or_desc": "Hello World"},
        {"id": "note", "type": "body", "text_or_desc": "Keep verbatim"},
        {"id": "ok_tag", "type": "tag", "text_or_desc": "Clean"},
    ],
    "design_notes": "Keep text legible.",
}


class RecordingLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete_json(self, messages, validator=None, *, max_tokens=None):
        self.calls.append(list(messages))
        reply = self.responses.pop(0)
        if validator is not None:
            validator(reply)
        return reply, {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}


def _full_layout(x_head: float = -10, fg_note: str = "#FFFFFF") -> dict:
    return {
        "canvas": {"width": 200, "height": 200, "bg_color": "#FFFFFF"},
        "version": 0,
        "elements": [
            {
                "id": "panel", "type": "card", "x": 0, "y": 50, "w": 200, "h": 150,
                "z_index": 0, "bg_color": "#111111",
            },
            {
                "id": "headline", "type": "title", "x": x_head, "y": 10, "w": 80, "h": 30,
                "z_index": 1, "text": "Hello World", "font_size": 24, "fg_color": "#111111",
            },
            {
                "id": "note", "type": "body", "x": 10, "y": 60, "w": 80, "h": 25,
                "z_index": 1, "text": "Keep verbatim", "font_size": 14, "fg_color": fg_note,
            },
            {
                "id": "ok_tag", "type": "tag", "x": 10, "y": 120, "w": 60, "h": 20,
                "z_index": 1, "text": "Clean", "font_size": 12, "fg_color": "#FFFFFF",
            },
        ],
    }


def _geom_patch(x_head: float = -10) -> dict:
    return {
        "updates": [
            {"id": "panel", "x": 0, "y": 50, "w": 200, "h": 150, "z_index": 0},
            {"id": "headline", "x": x_head, "y": 10, "w": 80, "h": 30, "z_index": 1},
            {"id": "note", "x": 10, "y": 60, "w": 80, "h": 25, "z_index": 1},
            {"id": "ok_tag", "x": 10, "y": 120, "w": 60, "h": 20, "z_index": 1},
        ]
    }


def _style_patch(fg_note: str = "#111111") -> dict:
    return {
        "updates": [
            {"id": "panel", "bg_color": "#111111"},
            {"id": "headline", "font_size": 24, "fg_color": "#111111"},
            {"id": "note", "font_size": 14, "fg_color": fg_note},
            {"id": "ok_tag", "font_size": 12, "fg_color": "#FFFFFF"},
        ]
    }


def _combined_patch(x_head: float = -10, fg_note: str = "#111111") -> dict:
    return {
        "updates": [
            {"id": "panel", "x": 0, "y": 50, "w": 200, "h": 150, "z_index": 0, "bg_color": "#111111"},
            {"id": "headline", "x": x_head, "y": 10, "w": 80, "h": 30, "z_index": 1, "font_size": 24, "fg_color": "#111111"},
            {"id": "note", "x": 10, "y": 60, "w": 80, "h": 25, "z_index": 1, "font_size": 14, "fg_color": fg_note},
            {"id": "ok_tag", "x": 10, "y": 120, "w": 60, "h": 20, "z_index": 1, "font_size": 12, "fg_color": "#FFFFFF"},
        ]
    }


def test_shared_format_templates_forbid_literal_field_key_and_concrete_hex_colors():
    for tpl in (
        OUTPUT_FORMAT_GEOMETRY_INIT,
        OUTPUT_FORMAT_PATCH_TEMPLATE,
        OUTPUT_FORMAT_BB_SINGLE_INIT_TEMPLATE,
    ):
        assert '"field"' not in tpl
        assert not re.search(r"#[0-9a-fA-F]{3,6}", tpl)
    assert "type 'image' and all text types are foreground (z_index >= 1)" in GEOMETRY_RULES


def test_all_four_detector_methods_share_verbatim_role_rules_without_asymmetry():
    forbidden = (
        "optional hex bg_color",
        "optional bg_color",
        "respect WCAG contrast",
        "ensure legible WCAG contrast",
        '"field":value',
    )

    # 0. CoLayout (Full)
    llm_co = RecordingLLM(
        _geom_patch(x_head=-10),
        _style_patch(fg_note="#111111"),
        {"updates": [{"id": "headline", "x": 10}]},
        {"updates": [{"id": "note", "fg_color": "#FFFFFF"}]},
    )
    rep_co = ours_full.run_method(BRIEF, llm_co, max_rounds=5)
    assert rep_co["metrics"]["CSR"] == 1.0
    co_r0_g, co_r0_s = llm_co.calls[0][0]["content"], llm_co.calls[1][0]["content"]
    co_rev_g, co_rev_s = llm_co.calls[2][0]["content"], llm_co.calls[3][0]["content"]
    assert GEOMETRY_RULES in co_r0_g
    assert STYLE_RULES in co_r0_s
    assert REVISION_RULES in co_rev_g and REVISION_RULES in co_rev_s
    for txt in (co_r0_g, co_r0_s, co_rev_g, co_rev_s):
        for phrase in forbidden:
            assert phrase not in txt

    # 1. Single-Agent + Det.
    llm_sad = RecordingLLM(_full_layout(x_head=-10, fg_note="#111111"), _full_layout(x_head=10, fg_note="#FFFFFF"))
    single_agent_detector.run_method(BRIEF, llm_sad, max_rounds=5)
    sad_r0_text = "\n".join(m["content"] for m in llm_sad.calls[0])
    sad_r1_text = "\n".join(m["content"] for m in llm_sad.calls[1])
    assert GEOMETRY_RULES in sad_r0_text and STYLE_RULES in sad_r0_text
    assert GEOMETRY_RULES in sad_r1_text and STYLE_RULES in sad_r1_text
    for phrase in forbidden:
        assert phrase not in sad_r0_text and phrase not in sad_r1_text

    # 2. Single-Agent + BB + Det.
    llm_bb = RecordingLLM(
        _combined_patch(x_head=-10, fg_note="#111111"),
        {"updates": [{"id": "headline", "x": 10}, {"id": "note", "fg_color": "#FFFFFF"}]},
    )
    single_agent_bb_detector.run_method(BRIEF, llm_bb, max_rounds=5)
    bb_r0_sys, bb_r1_sys = llm_bb.calls[0][0]["content"], llm_bb.calls[1][0]["content"]
    assert GEOMETRY_RULES in bb_r0_sys and STYLE_RULES in bb_r0_sys
    assert REVISION_RULES in bb_r1_sys
    for phrase in forbidden:
        assert phrase not in bb_r0_sys and phrase not in bb_r1_sys

    # 3. Single-Agent Two-Stage BB + Det.
    llm_ts = RecordingLLM(
        _geom_patch(x_head=-10),
        _style_patch(fg_note="#111111"),
        {"updates": [{"id": "headline", "x": 10}, {"id": "note", "fg_color": "#FFFFFF"}]},
    )
    sa_twostage_bb_det.run_method(BRIEF, llm_ts, max_rounds=5)
    ts_r0_g, ts_r0_s, ts_r1 = (
        llm_ts.calls[0][0]["content"],
        llm_ts.calls[1][0]["content"],
        llm_ts.calls[2][0]["content"],
    )
    assert GEOMETRY_RULES in ts_r0_g and STYLE_RULES in ts_r0_s
    assert REVISION_RULES in ts_r1
    for txt in (ts_r0_g, ts_r0_s, ts_r1):
        for phrase in forbidden:
            assert phrase not in txt


def test_sa_twostage_bb_det_twocall_init_unlocked_revision_tokens_and_arbiter():
    llm = RecordingLLM(
        _geom_patch(x_head=-10),
        _style_patch(fg_note="#111111"),
        {"updates": [{"id": "headline", "x": 10}, {"id": "note", "fg_color": "#FFFFFF"}]},
    )
    report = sa_twostage_bb_det.run_method(BRIEF, llm, max_rounds=5)
    assert report["rounds_used"] == 2
    assert report["total_tokens"] == 42
    assert report["metrics"]["CSR"] == 1.0
    assert report["history"][0]["init_stage"] == "geometry"
    assert report["history"][1]["init_stage"] == "style"
    assert all(e["role"] != "Priority Arbiter" for e in report["history"])

    r1_payload = json.loads(llm.calls[2][-1]["content"])
    assert r1_payload["allowed_ids"] == ["headline", "note"]
    by_id = {e["id"]: e for e in report["final_state"]["elements"]}
    assert by_id["headline"]["text"] == "Hello World"
    assert by_id["note"]["fg_color"] == "#FFFFFF"
    assert by_id["panel"]["bg_color"] == "#111111"
    assert by_id["ok_tag"]["x"] == 10.0
    assert _pre_arbiter_csr({**report, "method": "sa_twostage_bb_det"}, BRIEF) == 1.0

    # Strict rejection on nested "field" in sa_twostage_bb_det style init.
    with pytest.raises(ValueError, match="nested key 'field'"):
        sa_twostage_bb_det.run_method(
            BRIEF,
            RecordingLLM(_geom_patch(x_head=10), {"updates": [{"id": "panel", "field": {"bg_color": "#111111"}}]}),
            max_rounds=1,
        )

    # Exhausts K=1 and triggers Priority Arbiter; _pre_arbiter_csr returns 0.0.
    llm2 = RecordingLLM(
        _geom_patch(x_head=-10),
        _style_patch(fg_note="#111111"),
        {"updates": [{"id": "headline", "x": -5}]},
    )
    report2 = sa_twostage_bb_det.run_method(BRIEF, llm2, max_rounds=1)
    assert report2["rounds_used"] == 2
    assert report2["total_tokens"] == 42
    assert report2["metrics"]["CSR"] == 1.0
    assert report2["history"][-1]["role"] == "Priority Arbiter"
    assert _pre_arbiter_csr({**report2, "method": "sa_twostage_bb_det"}, BRIEF) == 0.0


def test_llm_client_seed_format_rejections_and_provider_metadata_capture():
    requests = []
    outputs = iter(['{"updates":[{"id":"a","field":{"x":1}}]}', '{"ok": true}'])

    def create(**request):
        requests.append(request)
        return SimpleNamespace(
            model="gpt-4o-mini-2024-07-18",
            system_fingerprint="fp_test_1234",
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(outputs)))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5, total_tokens=16),
        )

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = LLMClient(client=sdk, model="gpt-4o-mini-2024-07-18", seed=7)

    def validator(obj):
        if "ok" not in obj:
            raise ValueError("nested key 'field' not allowed")

    data, usage = client.complete_json([{"role": "user", "content": "Return JSON."}], validator=validator)
    assert data == {"ok": True}
    assert usage["total_tokens"] == 32
    assert client.format_rejections == 1
    assert requests[0]["model"] == "gpt-4o-mini-2024-07-18"
    assert requests[0]["seed"] == 7
    assert client.last_call_metadata["resolved_model"] == "gpt-4o-mini-2024-07-18"
    assert client.last_call_metadata["system_fingerprint"] == "fp_test_1234"
    assert client.last_call_metadata["format_rejections"] == 1

    with pytest.raises(ValueError, match="seed"):
        LLMClient(client=sdk, seed=True)


def test_runner_per_seed_directory_provenance_rejections_and_utc_timestamp(tmp_path, monkeypatch):
    replies = iter(["bad_json", '{"ok": true}', "bad1", "bad2", "bad3"])

    def make_client(*, model=None, seed=None):
        def create(**request):
            return SimpleNamespace(
                model=request["model"],
                system_fingerprint=f"fp_seed_{request.get('seed')}",
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))],
                usage=SimpleNamespace(prompt_tokens=8, completion_tokens=2, total_tokens=10),
            )

        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return LLMClient(client=sdk, model=model, seed=seed)

    monkeypatch.setattr(runner, "LLMClient", make_client)

    def stub_method(brief, llm, max_rounds):
        _, usage = llm.complete_json([{"role": "user", "content": "Return JSON."}])
        return {
            "final_state": {}, "history": [], "conflicts": [],
            "rounds_used": 1, "total_tokens": usage["total_tokens"],
            "metrics": dict.fromkeys(runner.METRICS, 0.0),
        }

    monkeypatch.setitem(runner.METHODS, "sa_twostage_bb_det", stub_method)
    summary = runner.run_experiments(
        ["sa_twostage_bb_det"], 1, 5, tmp_path,
        seed=2, model="gpt-4o-mini-2024-07-18", attempt_index=1,
    )
    assert summary == tmp_path / "seed_2" / "summary.csv"
    ckpt = json.loads((tmp_path / "seed_2" / "raw" / "sa_twostage_bb_det__poster_easy_01.json").read_bytes())
    assert ckpt["seed"] == 2
    assert ckpt["attempt_index"] == 1
    assert ckpt["effective_seed"] == 1002
    assert ckpt["model"] == "gpt-4o-mini-2024-07-18"
    assert ckpt["system_fingerprint"] == "fp_seed_1002"
    assert ckpt["resolved_models"] == ["gpt-4o-mini-2024-07-18"]
    assert ckpt["format_rejections"] == 1
    assert ckpt["timestamp_utc"].endswith("+00:00")

    monkeypatch.setitem(runner.METHODS, "single_agent", stub_method)
    with pytest.raises(InvalidCompletion):
        runner.run_experiments(
            ["single_agent"], 1, 5, tmp_path,
            seed=2, model="gpt-4o-mini-2024-07-18", attempt_index=2,
        )
    fail_rec = json.loads(next((tmp_path / "seed_2" / "failures").glob("*.json")).read_bytes())
    assert fail_rec["seed"] == 2
    assert fail_rec["attempt_index"] == 2
    assert fail_rec["effective_seed"] == 2002
    assert fail_rec["format_rejections"] == 3
    assert fail_rec["system_fingerprint"] == "fp_seed_2002"
    assert fail_rec["timestamp_utc"].endswith("+00:00")
