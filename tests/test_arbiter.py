import json

import pytest

from src.detector import evaluate_layout
from src.methods import ours_full, ours_no_detector
from src.methods.arbiter import resolve_conflicts
from src.schema import Canvas, Conflict, Element, LayoutState


def rect(id, x, y, w, h, kind="image", **style):
    return Element(id=id, type=kind, x=x, y=y, w=w, h=h, **style)


def layout(*elements):
    return LayoutState(canvas=Canvas(width=100, height=80, bg_color="#FFFFFF"),
                       elements=list(elements), version=5)


def test_clamps_oversized_background_container_as_well_as_foreground():
    original = layout(rect("card", -8, -15, 120, 110, "card", z_index=0),
                      rect("img", 82, 66, 20, 30))
    repaired = resolve_conflicts(original)
    assert [(e.x, e.y, e.w, e.h) for e in repaired.elements] == [
        (0, 0, 100, 80), (80, 50, 20, 30),
    ]
    assert original.elements[0].x == -8
    assert repaired.version == 6
    assert evaluate_layout(repaired)["conflicts"] == []


def test_nudges_on_smallest_displacement_axis_without_changing_other_boxes():
    original = layout(rect("a", 10, 10, 30, 30), rect("b", 38, 10, 30, 30),
                      rect("aside", 82, 60, 10, 10))
    repaired = resolve_conflicts(original)
    a, b, aside = repaired.elements
    assert (a.x, b.x) == (10, 40)
    assert (b.y, b.w, b.h) == (10, 30, 30)
    assert aside == original.elements[2]
    assert evaluate_layout(repaired)["metrics"]["CSR"] == 1


def test_tightly_packed_overlaps_are_positive_and_deterministic():
    original = layout(*(rect(str(i), 0, 0, 100, 80) for i in range(8)))
    repaired = resolve_conflicts(original)
    assert repaired == resolve_conflicts(original)
    assert repaired == resolve_conflicts(repaired)
    assert len(repaired.elements) == 8
    assert all(e.w > 0 and e.h > 0 for e in repaired.elements)
    assert evaluate_layout(repaired)["conflicts"] == []


@pytest.mark.parametrize("card_color,expected", [("#EEEEEE", "#000000"), ("#222222", "#FFFFFF")])
def test_repairs_inherited_contrast_after_geometry_changes(card_color, expected):
    original = layout(rect("card", 0, 0, 100, 80, "card", z_index=0, bg_color=card_color),
                      rect("body", 92, 77, 18, 18, "body", fg_color=card_color, text="Read"),
                      rect("other", 70, 60, 15, 16))
    repaired = resolve_conflicts(original)
    assert evaluate_layout(repaired)["conflicts"] == []
    assert repaired.elements[1].fg_color == expected
    assert repaired.elements[1].text == "Read"
    assert repaired.elements[0] == original.elements[0]


def test_ablation_only_repairs_issues_actually_reported_by_its_critic():
    original = layout(rect("outside", -10, 0, 10, 10),
                      rect("unreported", 30, 30, 10, 10, "title", fg_color="#FFFFFF"))
    reported = Conflict(conflict_type="overflow", element_ids=["outside"], severity="hard",
                        detail="Left edge outside", numeric_value=10)
    repaired = resolve_conflicts(original, [reported])
    assert repaired.elements[0].x == 0
    assert repaired.elements[1] == original.elements[1]
    assert [c.conflict_type for c in evaluate_layout(repaired)["conflicts"]] == ["contrast"]


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete_json(self, messages, validator=None):
        self.requests.append(json.loads(messages[-1]["content"]))
        response = next(self.responses)
        validator(response)
        return response, {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}


def broken_start():
    brief = {
        "canvas_width": 100, "canvas_height": 80, "canvas_bg_color": "#FFFFFF",
        "required_elements": [
            {"id": "a", "type": "body", "text_or_desc": "Read me"},
            {"id": "b", "type": "image", "text_or_desc": "Forest"},
        ],
        "design_notes": "Place the text next to the image.",
    }
    geometry = {"updates": [
        {"id": name, "x": 0, "y": 10, "w": 20, "h": 20, "z_index": 1} for name in ("a", "b")
    ]}
    style = {"updates": [
        {"id": "a", "fg_color": "#EEEEEE", "font_size": 16}, {"id": "b", "bg_color": None},
    ]}
    return brief, geometry, style


def test_full_routes_typed_reports_then_applies_fallback_after_unproductive_round():
    brief, geometry, style = broken_start()
    llm = ScriptedClient([geometry, style, {"updates": []}, {"updates": []}])
    result = ours_full.run_method(brief, llm, max_rounds=1)
    assert result["metrics"]["CSR"] == 1
    assert [c["conflict_type"] for c in llm.requests[2]["conflicts"]] == ["overlap"]
    assert [c["conflict_type"] for c in llm.requests[3]["conflicts"]] == ["contrast"]
    assert llm.requests[3]["allowed_ids"] == ["a"]
    assert result["history"][-1]["role"] == "Priority Arbiter"
    assert result["rounds_used"] == 2 and result["total_tokens"] == 12
    assert result["final_state"]["version"] == 2


def test_no_detector_honors_raw_critic_false_negative_even_when_posthoc_metrics_fail():
    brief, geometry, style = broken_start()
    llm = ScriptedClient([geometry, style, {"conflicts": []}])
    result = ours_no_detector.run_method(brief, llm)
    assert set(llm.requests[-1]) == {"brief", "state"}
    assert result["metrics"]["CSR"] == 0
    assert result["rounds_used"] == 1 and result["total_tokens"] == 9
    assert result["history"][-1]["role"] == "Compliance Critic"
