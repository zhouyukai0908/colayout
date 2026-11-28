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

STRICT_GEOMETRY_RESPONSE = {"updates": [
    {"id": "title", "x": 10, "y": 10, "w": 260, "h": 50, "z_index": 1},
    {"id": "photo", "x": 10, "y": 70, "w": 260, "h": 100, "z_index": 1},
    {"id": "panel", "x": 0, "y": 0, "w": 320, "h": 180, "z_index": 0},
]}

STRICT_STYLE_RESPONSE = {"updates": [
    {"id": "title", "font_size": 25, "fg_color": "#222222"},
    {"id": "photo", "bg_color": None},
    {"id": "panel", "bg_color": "#F5F5F5"},
]}


def test_initial_geometry_does_not_prefill_passing_colors_and_requires_all_fields():
    state = _initial_geometry(STRICT_GEOMETRY_RESPONSE, BRIEF)
    title, photo, panel = state.elements
    assert (title.text, title.font_size, title.fg_color, title.bg_color) == ("River walks", None, None, None)
    assert (photo.bg_color, panel.bg_color, panel.z_index) == (None, None, 0)

    # Incomplete geometry must raise instead of silently falling back to canvas defaults.
    with pytest.raises(ValueError, match="Initial geometry"):
        _initial_geometry({"updates": [{"id": "photo", "x": 12}]}, BRIEF)


def test_legacy_replay_mode_still_supports_pre_v2_logs():
    legacy_resp = {"updates": [{"id": "photo", "x": 12}, {"id": "title", "text": "wrong"}]}
    state = _initial_geometry(legacy_resp, BRIEF, strict_keys=False)
    title, photo, panel = state.elements
    assert [e.id for e in state.elements] == ["title", "photo", "panel"]
    assert (title.text, title.x, title.y, title.w, title.h) == ("River walks", 0, 0, 320, 180)
    assert (title.font_size, title.fg_color) == (16, "#000000")
    assert (photo.x, photo.y, photo.z_index, panel.z_index) == (12, 0, 1, 0)


def test_empty_initial_proposal_still_needs_an_id():
    with pytest.raises(ValueError, match="element id"):
        _initial_geometry({"updates": []}, BRIEF)


def test_strict_two_stage_init_preserves_copy_and_requires_fg_and_card_bg():
    class Client:
        def __init__(self):
            self.responses = iter([STRICT_GEOMETRY_RESPONSE, STRICT_STYLE_RESPONSE])
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
    assert photo["type"] == "image" and panel["bg_color"] == "#F5F5F5" and panel["z_index"] == 0


@pytest.mark.parametrize("fields,row,pattern", [
    (GEOMETRY, {"id": "title", "font_size": 20}, "unknown or non-writable"),
    (GEOMETRY, {"id": "title", "type": "tag"}, "unknown or non-writable"),
    (STYLE, {"id": "title", "x": 10}, "unknown or non-writable"),
    (STYLE, {"id": "title", "field": {"fg_color": "#111111"}}, "nested key 'field'"),
    (STYLE, {"id": "title", "fg_color": {"hex": "#111111"}}, "nested object"),
    (STYLE, {"id": "title"}, "at least one field"),
    (STYLE, {"id": "photo", "fg_color": "#111111"}, "allowed_ids"),
])
def test_strict_updates_never_silently_drops_unknown_nested_or_out_of_scope_rows(fields, row, pattern):
    original = _initial_geometry(STRICT_GEOMETRY_RESPONSE, BRIEF)
    with pytest.raises(ValueError, match=pattern):
        _patched({"updates": [row]}, original, BRIEF, {"title"}, fields)


@pytest.mark.parametrize("style_rows,pattern", [
    ([{"id": "title", "font_size": 20}, {"id": "panel", "bg_color": "#222222"}], "missing fg_color"),
    ([{"id": "title", "fg_color": "#222222"}, {"id": "panel", "bg_color": "#222222"}], "missing font_size"),
    ([{"id": "title", "font_size": 20, "fg_color": "#222222"}, {"id": "panel", "bg_color": None}], "missing non-null bg_color"),
    ([{"id": "title", "font_size": 20, "fg_color": "#222222"}], "missing non-null bg_color"),
])
def test_initial_style_requires_text_fg_font_size_and_card_bg(style_rows, pattern):
    original = _initial_geometry(STRICT_GEOMETRY_RESPONSE, BRIEF)
    all_ids = {e.id for e in original.elements}
    with pytest.raises(ValueError, match=pattern):
        _patched({"updates": style_rows}, original, BRIEF, all_ids, STYLE, require_initial_style=True)


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
    original = _initial_geometry(STRICT_GEOMETRY_RESPONSE, BRIEF)
    with pytest.raises(ValueError):
        _patched({"updates": [row]}, original, BRIEF, {"title"}, fields)
