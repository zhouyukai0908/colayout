import pytest

from src.detector import evaluate_layout
from src.schema import Canvas, Conflict, Element, LayoutState


def element(name, x, y, w=10, h=10, kind="image", **style):
    return Element(id=name, type=kind, x=x, y=y, w=w, h=h, **style)


def layout(*elements):
    return LayoutState(
        canvas=Canvas(width=100, height=100, bg_color="#FFFFFF"),
        elements=list(elements),
    )


def test_clean_layout_on_background_card():
    state = layout(
        element("panel", 0, 0, 100, 100, "card", z_index=0, bg_color="#171717"),
        element("heading", 10, 10, 60, 20, "title", text="CoLayout", fg_color="#FFFFFF"),
        element("photo", 10, 30, 80, 70),
    )
    report = evaluate_layout(state)
    assert report["conflicts"] == []
    assert report["metrics"] == {"OR": 0, "BOR": 0, "CPR": 1, "AS": 1, "CSR": 1}


def test_pairwise_overlap_uses_foreground_area_and_excludes_touching_edges():
    state = layout(
        element("background", 0, 0, 100, 100, z_index=0),
        element("a", 0, 0),
        element("b", 5, 0, w=20),
        element("c", 10, 0),
    )
    report = evaluate_layout(state)
    assert report == evaluate_layout(state)
    assert report["metrics"]["OR"] == pytest.approx((50 + 100) / 400)
    assert report["metrics"]["CSR"] == 0
    conflicts = report["conflicts"]
    assert len(conflicts) == 2
    for conflict in conflicts:
        assert isinstance(conflict, Conflict)
        assert conflict.conflict_type == "overlap" and conflict.severity == "hard"
        assert conflict.detail
    assert {frozenset(c.element_ids): c.numeric_value for c in conflicts} == {
        frozenset(("a", "b")): 50,
        frozenset(("b", "c")): 100,
    }


def test_common_intersection_counts_for_each_pair():
    report = evaluate_layout(layout(*(element(str(i), 0, 0) for i in range(4))))
    assert len(report["conflicts"]) == 6
    assert report["metrics"]["OR"] == 1.5


def test_overflow_counts_each_element_once_on_all_four_boundaries():
    state = layout(
        element("left", -1, 30),
        element("top", 30, -0.1),
        element("right", 92, 30),
        element("bottom", 40, 91),
        element("bottom_right", 99, 99, kind="card", z_index=0),
        element("inside", 60, 90),
    )
    report = evaluate_layout(state)
    assert report["metrics"]["BOR"] == pytest.approx(5 / 6)
    assert report["metrics"]["CSR"] == 0
    assert sorted(
        (c.element_ids, c.conflict_type, c.severity) for c in report["conflicts"]
    ) == [
        ([name], "overflow", "hard")
        for name in ("bottom", "bottom_right", "left", "right", "top")
    ]


@pytest.mark.parametrize("kind", ["title", "subtitle", "body", "button", "tag"])
def test_wcag_threshold_changes_at_24_pixels_without_rounding(kind):
    state = layout(
        element("normal", 5, 0, kind=kind, font_size=23, fg_color="#777777", text="a"),
        element("large", 5, 20, kind=kind, font_size=24, fg_color="#777777", text="b"),
        element("faint", 5, 40, kind=kind, font_size=24, fg_color="#959595", text="c"),
    )
    report = evaluate_layout(state)
    assert report["metrics"]["CPR"] == pytest.approx(1 / 3)
    assert report["metrics"]["CSR"] == 0
    failures = {c.element_ids[0]: c for c in report["conflicts"]}
    assert set(failures) == {"normal", "faint"}
    assert all(c.conflict_type == "contrast" and c.severity == "hard" for c in failures.values())
    assert failures["normal"].numeric_value == pytest.approx(4.47809, abs=1e-5)
    assert failures["faint"].numeric_value == pytest.approx(2.99535, abs=1e-5)


def test_contrast_background_precedence_and_card_paint_order():
    state = layout(
        element("base", 0, 0, 80, 100, "card", z_index=0, bg_color="#FFFFFF"),
        element("dark", 0, 0, 80, 100, "card", z_index=0, bg_color="#000000"),
        element("inherited", 0, 0, kind="body", text="Black on black"),
        element("explicit", 0, 20, kind="tag", bg_color="#FFFFFF", text="Black on white"),
        element("outside", 90, 20, kind="button", text="Canvas"),
        element("above", 0, 0, 80, 100, "card", z_index=2, bg_color="#FFFFFF"),
    )
    report = evaluate_layout(state)
    assert report["metrics"]["CPR"] == pytest.approx(2 / 3)
    conflict, = report["conflicts"]
    assert (conflict.element_ids, conflict.conflict_type, conflict.numeric_value) == (
        ["inherited"], "contrast", 1,
    )


@pytest.mark.parametrize(
    "peer_x, peer_w, expected",
    [
        (12, 30, 2 / 3),      # Left edges differ by exactly 2 px.
        (20, 12, 2 / 3),      # Right edges at 30 and 32.
        (15, 10, 2 / 3),      # Centers at 20.
        (12.01, 30, 0),       # Closest corresponding anchors exceed tolerance.
    ],
)
def test_alignment_is_a_fraction_of_elements_and_independent_of_csr(peer_x, peer_w, expected):
    state = layout(
        element("first", 10, 0, w=20),
        element("peer", peer_x, 30, w=peer_w),
        element("isolated", 80, 60),
    )
    report = evaluate_layout(state)
    assert report["metrics"]["AS"] == pytest.approx(expected)
    assert report["metrics"]["CSR"] == 1
    assert report["conflicts"] == []


def test_empty_layout():
    report = evaluate_layout(layout())
    assert report["conflicts"] == []
    assert report["metrics"] == {"OR": 0, "BOR": 0, "CPR": 1, "AS": 0, "CSR": 1}
