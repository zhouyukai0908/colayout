import json

import pytest

from src.methods import (
    common, naive_multi_agent, self_refine, single_agent,
    single_agent_bb_detector, single_agent_detector,
)


BRIEF = {
    "task_id": "example", "domain": "poster", "difficulty": "easy",
    "canvas_width": 100, "canvas_height": 100, "canvas_bg_color": "#FFFFFF",
    "required_elements": [{"id": "title", "type": "title", "text_or_desc": "Hello"}],
    "design_notes": "Use an inset header.",
}


def layout(x=5):
    return {
        "canvas": {"width": 100, "height": 100, "bg_color": "#ffffff"},
        "elements": [{"id": "title", "type": "title", "x": x, "y": 5,
                      "w": 20, "h": 20, "text": "Hello"}],
    }


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.completion_limits = []

    def complete_json(self, messages, validator=None, *, max_tokens=None):
        self.calls.append(list(messages))
        self.completion_limits.append(max_tokens)
        reply = self.responses.pop(0)
        validator(reply)
        return reply, {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


def test_single_agent_one_round_and_serializable_report():
    llm = ScriptedLLM(layout())
    report = single_agent.run_method(BRIEF, llm)
    assert len(llm.calls) == report["rounds_used"] == 1
    assert report["total_tokens"] == 7
    assert json.loads(json.dumps(report)) == report


def test_self_refine_revises_then_follows_its_own_stop_signal(monkeypatch):
    llm = ScriptedLLM(
        layout(), {"status": "REVISED", "critique": "Move the header.", "layout": layout(-5)},
        {"status": "NO_CONFLICTS"},
    )
    # Even an erroneous stop decision stands. The detector first sees the final state.
    calls_before_scoring = []
    evaluate = common.evaluate_layout

    def record_score(state):
        calls_before_scoring.append(len(llm.calls))
        return evaluate(state)

    monkeypatch.setattr(common, "evaluate_layout", record_score)
    report = self_refine.run_method(BRIEF, llm, max_rounds=5)
    assert calls_before_scoring == [3]
    assert report["rounds_used"] == 3
    assert report["total_tokens"] == 21
    assert report["conflicts"][0]["conflict_type"] == "overflow"
    assert '"x":-5.0' in llm.calls[2][-1]["content"]


def test_naive_agents_exchange_full_conversation_until_budget(monkeypatch):
    proposal = {"message": "Keep the caption above the image.", "layout": layout()}
    revision = {"message": "Use neutral colors.", "layout": layout(8)}
    critique = {"message": "Increase the left inset.", "status": "REVISE"}
    llm = ScriptedLLM(proposal, revision, critique, proposal, revision, critique)
    evaluate = common.evaluate_layout
    score_calls = []

    def record_score(state):
        score_calls.append(len(llm.calls))
        return evaluate(state)

    monkeypatch.setattr(common, "evaluate_layout", record_score)
    report = naive_multi_agent.run_method(BRIEF, llm, max_rounds=2)
    assert score_calls == [6]
    assert report["rounds_used"] == 2
    assert report["total_tokens"] == 42
    assert report["final_state"]["elements"][0]["x"] == 8
    assert [entry["role"] for entry in report["history"]] == [
        "Layout Architect", "Visual & Typography Agent", "Compliance Critic",
    ] * 2
    for before, after in zip(llm.calls, llm.calls[1:]):
        old_chat = [m for m in before if m["role"] == "assistant"]
        new_chat = [m for m in after if m["role"] == "assistant"]
        assert new_chat[:-1] == old_chat
    round_two_input = json.dumps(llm.calls[3])
    for response in (proposal, revision, critique):
        assert response["message"] in round_two_input


@pytest.mark.parametrize("method", [single_agent, self_refine, naive_multi_agent])
def test_zero_round_budget_does_not_issue_a_request(method):
    with pytest.raises(ValueError, match="max_rounds"):
        method.run_method(BRIEF, ScriptedLLM(), max_rounds=0)


def test_cannot_drop_required_copy_to_evade_a_constraint():
    proposal = layout()
    proposal["elements"][0]["text"] = ""
    with pytest.raises(ValueError, match="verbatim"):
        single_agent.run_method(BRIEF, ScriptedLLM(proposal))


def test_naive_accepts_bare_and_wrapped_layouts_and_normalizes_only_foreground():
    brief = {
        **BRIEF, "required_elements": [
            *BRIEF["required_elements"], {"id": "panel", "type": "card", "text_or_desc": "Paper"},
        ],
    }
    proposal = layout()
    proposal["elements"][0]["z_index"] = -2
    card = {"id": "panel", "type": "card", "x": 0, "y": 0, "w": 100, "h": 100, "z_index": 0}
    proposal["elements"].append(card)
    visual = layout(6)
    visual["elements"][0]["z_index"] = 0
    visual["elements"].append(card)

    llm = ScriptedLLM(proposal, {"layout": visual}, {"status": "NO_CONFLICTS"})
    report = naive_multi_agent.run_method(brief, llm)
    assert len(llm.calls) == 3
    assert [e["z_index"] for e in report["final_state"]["elements"]] == [1, 0]
    for entry in report["history"][:2]:
        elements = entry["response"]["layout"]["elements"]
        assert [e["z_index"] for e in elements] == [1, 0]
        assert entry["response"]["message"]
    architect, visual_limit, critic = llm.completion_limits
    assert 0 < critic < architect == visual_limit <= 4096
    assert '"z_index":1' in llm.calls[1][-1]["content"]
    assert report["total_tokens"] == 21
    assert proposal["elements"][0]["z_index"] == -2


def test_naive_messages_are_bounded_and_original_turns_stay_in_chat():
    message = " ".join(f"word{i}" for i in range(150))
    proposal = {"message": message, "layout": layout()}
    llm = ScriptedLLM(proposal, proposal, {"message": message, "status": "NO_CONFLICTS"})
    report = naive_multi_agent.run_method(BRIEF, llm)
    for entry in report["history"]:
        words = entry["response"]["message"].split()
        assert words == message.split()[:120]
    assert "word119" in llm.calls[2][-2]["content"]
    assert "word120" not in str(llm.calls[2])
    assert len([m for m in llm.calls[2] if m["role"] == "assistant"]) == 2


def test_naive_keeps_validation_errors_for_invalid_card_and_missing_required_copy():
    brief = {**BRIEF, "required_elements": [{"id": "panel", "type": "card", "text_or_desc": "Paper"}]}
    invalid = layout()
    invalid["elements"][0].update(id="panel", type="card", z_index=1)
    with pytest.raises(ValueError, match="z_index") as caught:
        naive_multi_agent.run_method(brief, ScriptedLLM(invalid))
    assert caught.value.colayout_role == naive_multi_agent.ARCHITECT
    assert caught.value.colayout_round == 1
    with pytest.raises(ValueError, match="verbatim"):
        invalid_copy = layout()
        del invalid_copy["elements"][0]["text"]
        naive_multi_agent.run_method(BRIEF, ScriptedLLM({"layout": invalid_copy}))


def test_uncapped_method_remains_compatible_with_client_without_max_tokens_argument():
    class LegacyClient:
        def complete_json(self, messages, validator=None):
            response = layout()
            validator(response)
            return response, {"total_tokens": 7, "prompt_tokens": 5, "completion_tokens": 2}

    assert single_agent.run_method(BRIEF, LegacyClient())["total_tokens"] == 7


def test_naive_shares_only_latest_layout_and_preserves_all_discussion_across_keep_turns():
    llm = ScriptedLLM(
        {"layout": layout(15.5), "message": "Inset the title."},
        {"layout": layout(23.5), "message": "Move the title farther right."},
        {"status": "REVISE", "message": "Check the remaining space."},
        {"action": "KEEP_LAYOUT", "message": "The revised inset fits."},
        {"status": "KEEP_LAYOUT", "message": "The existing color is legible."},
        {"status": "NO_CONFLICTS", "message": "Accept the current draft."},
    )
    result = naive_multi_agent.run_method(BRIEF, llm, max_rounds=4)
    assert result["rounds_used"] == 2
    assert result["total_tokens"] == 42
    assert result["final_state"]["elements"][0]["x"] == 23.5
    for index, request in enumerate(llm.calls):
        documents = [m["content"].split("\n", 1)[1] for m in request
                     if m["content"].startswith("Latest complete proposal:\n")]
        expected_x = [] if index == 0 else [15.5 if index == 1 else 23.5]
        assert [json.loads(doc)["elements"][0]["x"] for doc in documents] == expected_x
        assert all('"canvas":' not in m["content"] for m in request if m["role"] == "assistant")
        if index >= 2:
            assert all("15.5" not in m["content"] for m in request)
        discussion = [m["content"] for m in request if m["role"] == "assistant"]
        assert len(discussion) == index
        for entry, text in zip(result["history"], discussion):
            assert entry["response"]["message"] in text
    for index in (3, 4):
        assert result["history"][index]["response"]["action"] == "KEEP_LAYOUT"
        assert "layout" not in result["history"][index]["response"]


def test_naive_requires_initial_document_before_accepting_keep_layout():
    with pytest.raises(ValueError, match="first editor must provide a complete layout"):
        naive_multi_agent.run_method(BRIEF, ScriptedLLM({"action": "KEEP_LAYOUT"}))


def test_single_agent_detector_repairs_from_deterministic_conflicts_and_falls_back_to_arbiter():
    # Case 1: resolves overflow in 1 correction round without arbiter.
    llm = ScriptedLLM(layout(-5), layout(10))
    report = single_agent_detector.run_method(BRIEF, llm, max_rounds=5)
    assert report["rounds_used"] == 2
    assert report["metrics"]["CSR"] == 1.0
    assert "overflow" in llm.calls[1][-1]["content"]
    assert all(entry["role"] != "Priority Arbiter" for entry in report["history"])

    # Case 2: exhausts correction budget and invokes Priority Arbiter.
    llm2 = ScriptedLLM(layout(-5), layout(-4), layout(-3))
    report2 = single_agent_detector.run_method(BRIEF, llm2, max_rounds=2)
    assert report2["rounds_used"] == 3
    assert report2["metrics"]["CSR"] == 1.0
    assert report2["history"][-1]["role"] == "Priority Arbiter"
    assert report2["final_state"]["elements"][0]["x"] == 0.0


def test_single_agent_bb_detector_programmatic_copy_union_allowed_ids_and_arbiter():
    from src.export_results import _pre_arbiter_csr

    brief = {
        "task_id": "bb_test", "domain": "poster", "difficulty": "easy",
        "canvas_width": 200, "canvas_height": 200, "canvas_bg_color": "#FFFFFF",
        "required_elements": [
            {"id": "headline", "type": "title", "text_or_desc": "Hello World"},
            {"id": "note", "type": "body", "text_or_desc": "Keep verbatim"},
            {"id": "ok_tag", "type": "tag", "text_or_desc": "Clean"},
        ],
        "design_notes": "Keep text legible.",
    }
    # Round 0: omit text (injected programmatically), headline overflows, note has white-on-white contrast fail,
    # ok_tag is clean (must NOT appear in round-1 allowed_ids).
    r0 = {"updates": [
        {"id": "headline", "x": -10, "y": 10, "w": 80, "h": 30, "z_index": 1, "font_size": 24, "fg_color": "#111111"},
        {"id": "note", "x": 10, "y": 60, "w": 80, "h": 25, "z_index": 1, "font_size": 14, "fg_color": "#FFFFFF"},
        {"id": "ok_tag", "x": 10, "y": 120, "w": 60, "h": 20, "z_index": 1, "font_size": 12, "fg_color": "#000000"},
    ]}
    # Round 1: fix BOTH geometry (headline x) and style (note fg_color) in one partial patch;
    # attempt to touch non-allowed ok_tag and overwrite text -> both must be ignored!
    r1 = {"updates": [
        {"id": "headline", "x": 10, "text": "TAMPERED"},
        {"id": "note", "fg_color": "#111111"},
        {"id": "ok_tag", "x": 99},
    ]}
    llm = ScriptedLLM(r0, r1)
    report = single_agent_bb_detector.run_method(brief, llm, max_rounds=5)
    assert report["rounds_used"] == 2
    assert report["metrics"]["CSR"] == 1.0
    assert all(e["role"] != "Priority Arbiter" for e in report["history"])
    # Check exact union of allowed_ids passed to Round 1 prompt
    r1_payload = json.loads(llm.calls[1][-1]["content"])
    assert r1_payload["allowed_ids"] == ["headline", "note"]
    by_id = {e["id"]: e for e in report["final_state"]["elements"]}
    assert by_id["headline"]["text"] == "Hello World"
    assert by_id["note"]["text"] == "Keep verbatim"
    assert by_id["ok_tag"]["x"] == 10.0  # untouched because not in allowed_ids
    assert _pre_arbiter_csr({**report, "method": "single_agent_bb_detector"}, brief) == 1.0

    # Case 2: exhausts K=1 and falls back to Priority Arbiter; _pre_arbiter_csr must replay 0.0
    llm2 = ScriptedLLM(r0, {"updates": [{"id": "headline", "x": -5}]})
    report2 = single_agent_bb_detector.run_method(brief, llm2, max_rounds=1)
    assert report2["rounds_used"] == 2
    assert report2["metrics"]["CSR"] == 1.0
    assert report2["history"][-1]["role"] == "Priority Arbiter"
    assert _pre_arbiter_csr({**report2, "method": "single_agent_bb_detector"}, brief) == 0.0
