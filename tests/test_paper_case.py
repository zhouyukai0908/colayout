import pytest
from PIL import Image

from src.detector import evaluate_layout
from src.paper_case import PANELS, export_case
from src.schema import LayoutState


def _records(task, *, difficulty="hard", offset=2):
    records = {}
    for method in PANELS:
        state = LayoutState.model_validate({
            "canvas": {"width": 120, "height": 80, "bg_color": "#FFFFFF"},
            "elements": [
                {"id": "a", "type": "body", "text": "Research", "x": 2, "y": 2, "w": 64, "h": 24},
                {"id": "b", "type": "button", "text": "Join us", "x": 2,
                 "y": offset if method == "single_agent" else 33, "w": 60, "h": 24},
            ],
        })
        report = evaluate_layout(state)
        records[method, task] = {
            "method": method, "task_id": task, "difficulty": difficulty,
            "final_state": state.model_dump(), "metrics": report["metrics"],
            "conflicts": [c.model_dump() for c in report["conflicts"]],
        }
    return records


def test_selects_visible_hard_case_and_writes_both_formats(tmp_path):
    records = _records("a", difficulty="easy") | _records("b", offset=150) | _records("c")
    assert export_case(records, None, tmp_path, mock=True) == "c"
    png = tmp_path / "fig3_case_study.png"
    with Image.open(png) as image:
        assert image.info["dpi"][0] == pytest.approx(300, abs=0.01)
        assert image.width > image.height
    assert (tmp_path / "fig3_case_study.pdf").read_bytes().startswith(b"%PDF-")
    assert export_case(records, "a", tmp_path) == "a"


@pytest.mark.parametrize("corruption", ["OR", "conflicts"])
def test_rejects_stale_checkpoint(tmp_path, corruption):
    records = _records("t")
    if corruption == "conflicts":
        records["single_agent", "t"]["conflicts"] = []
    else:
        records["single_agent", "t"]["metrics"]["OR"] = 1
    with pytest.raises(ValueError, match="saved .* differs? from"):
        export_case(records, "t", tmp_path)


def test_rejects_case_with_no_visible_difference(tmp_path):
    with pytest.raises(ValueError, match="visible baseline hard conflict"):
        export_case(_records("t", offset=150), None, tmp_path)
    assert not list(tmp_path.iterdir())


def test_explicit_case_can_show_three_clean_outputs(tmp_path):
    assert export_case(_records("t", offset=33), "t", tmp_path) == "t"
