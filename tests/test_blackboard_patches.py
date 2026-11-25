import pytest

from src.methods.blackboard import GEOMETRY, STYLE, _initial_geometry, _patched
from src.methods.ours_full import run_method


BRIEF = {
    "canvas_width": 320,
    "canvas_height": 180,
    "canvas_bg_color": "#FFFFFF",
    "required_elements": [
        {"id": "title", "type": "title", "text_or_desc": "River walks"},
        {"id": "photo", "type": "image", "text_or_desc": "Riverside trail"},
        {"id": "panel", "type": "card", "text_or_desc": "Background"},
    ],
    "design_notes": "Align the title and image.",
}

GEOMETRY_RESPONSE = {"updates": [
    {"id": "title", "x": 10, "y": 10, "w": 260, "h": 50, "z_index": 1,
     "type": "button", "text": "wrong", "font_size": "ignored", "bg_color": "ignored"},
    {"id": "photo", "x": 10, "y": 70, "w": 260, "h": 100},
    {"id": "panel", "type": "card"},
]}


def test_read_only_row_and_omitted_geometry_get_valid_initial_defaults():
    response = {"updates": [{"id": "photo", "x": 12}, {"id": "title", "text": "wrong"}]}
    state = _initial_geometry(response, BRIEF)
    title, photo, panel = state.elements
    assert [e.id for e in state.elements] == ["title", "photo", "panel"]
    assert (title.text, title.x, title.y, title.w, title.h) == ("River walks", 0, 0, 320, 180)
    assert (title.font_size, title.fg_color) == (16, "#000000")
    assert (photo.x, photo.y, photo.z_index, panel.z_index) == (12, 0, 1, 0)


def test_empty_initial_proposal_still_needs_an_id():
    with pytest.raises(ValueError, match="element id"):
        _initial_geometry({"updates": []}, BRIEF)


def test_mixed_read_only_and_owned_rows_need_one_call_per_initial_role():
    class Client:
        def __init__(self):
            self.responses = iter([
                GEOMETRY_RESPONSE,
                {"updates": [
                    {"id": "title", "x": 298, "y": "wrong", "type": "tag",
                     "text": "wrong", "font_size": 25},
                    {"id": "title", "fg_color": "#222222"},
                    {"id": "unknown", "bg_color": "ignored"},
                    {"id": "photo", "type": "body"},
                ]},
            ])
            self.calls = 0

        def complete_json(self, messages, validator=None):
            self.calls += 1
            response = next(self.responses)
            validator(response)
            return response, {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}

    client = Client()
    result = run_method(BRIEF, client)
    assert client.calls == 2
    assert result["total_tokens"] == 6 and result["conflicts"] == []
    title, photo, panel = result["final_state"]["elements"]
    assert (title["type"], title["text"], title["x"], title["y"]) == ("title", "River walks", 10, 10)
    assert (title["font_size"], title["fg_color"], title["bg_color"]) == (25, "#222222", None)
    assert photo["type"] == "image" and panel["z_index"] == 0


@pytest.mark.parametrize("fields,row", [
    (GEOMETRY, {"id": "title", "font_size": "ignored", "type": "tag"}),
    (STYLE, {"id": "title", "x": "ignored", "y": -500}),
    (STYLE, {"id": "title"}),
    (STYLE, {"id": "photo", "fg_color": "invalid value outside target scope"}),
])
def test_filtered_noop_leaves_elements_and_version_unchanged(fields, row):
    original = _initial_geometry(GEOMETRY_RESPONSE, BRIEF)
    updated = _patched({"updates": [row]}, original, BRIEF, {"title"}, fields)
    assert updated == original


@pytest.mark.parametrize("fields,row", [
    (STYLE, {"fg_color": "#FFFFFF"}),
    (STYLE, {"id": "", "font_size": 14}),
    (STYLE, {"id": "title", "font_size": "25"}),
    (STYLE, {"id": "title", "fg_color": "navy"}),
    (GEOMETRY, {"id": "title", "x": [12]}),
    (GEOMETRY, {"id": "title", "w": 0}),
    (GEOMETRY, {"id": "title", "z_index": 0}),
])
def test_missing_id_and_invalid_owned_values_still_fail_validation(fields, row):
    original = _initial_geometry(GEOMETRY_RESPONSE, BRIEF)
    with pytest.raises(ValueError):
        _patched({"updates": [row]}, original, BRIEF, {"title"}, fields)
