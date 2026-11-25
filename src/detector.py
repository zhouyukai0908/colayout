"""Deterministic geometry and WCAG 2.1 checks for a shared layout snapshot."""

from itertools import combinations

from .schema import Conflict, Element, LayoutState


TEXT_TYPES = frozenset({"title", "subtitle", "body", "button", "tag"})
ALIGNMENT_TOLERANCE = 2.0


def effective_background(state: LayoutState, element_index: int) -> str:
    """Resolve a fill using fully enclosing cards below the element in paint order."""
    element = state.elements[element_index]
    if element.bg_color is not None:
        return element.bg_color

    color = state.canvas.bg_color
    element_order = (element.z_index, element_index)
    best_order = (-1, -1)
    for index, card in enumerate(state.elements):
        order = (card.z_index, index)
        if card.type != "card" or card.bg_color is None or not best_order < order < element_order:
            continue
        if (
            card.x <= element.x
            and card.y <= element.y
            and element.x + element.w <= card.x + card.w
            and element.y + element.h <= card.y + card.h
        ):
            best_order = order
            color = card.bg_color
    return color


def _luminance(color: str) -> float:
    """Relative luminance of an opaque sRGB color normalized by the schema."""
    def linear_channel(start: int) -> float:
        channel = int(color[start:start + 2], 16) / 255.0
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    return 0.2126 * linear_channel(1) + 0.7152 * linear_channel(3) + 0.0722 * linear_channel(5)


def _intersect_area(a: Element, b: Element) -> float:
    width = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    height = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    return max(0.0, width) * max(0.0, height)


def _anchors(element: Element) -> tuple[float, float, float]:
    return element.x, element.x + element.w, element.x + element.w / 2


def _foreground_metrics(elements: list[Element], conflicts: list[Conflict]) -> tuple[float, float]:
    foreground = [e for e in elements if e.z_index >= 1 and e.type != "card"]
    total_area = sum(e.w * e.h for e in foreground)
    overlap_area = 0.0
    aligned = set()
    for a, b in combinations(foreground, 2):
        area = _intersect_area(a, b)
        if area > 0:
            overlap_area += area
            conflicts.append(Conflict(
                conflict_type="overlap", element_ids=[a.id, b.id], severity="hard",
                detail=f"Intersection area is {area:g} square pixels.", numeric_value=area,
            ))
        if any(abs(x - y) <= ALIGNMENT_TOLERANCE for x, y in zip(_anchors(a), _anchors(b))):
            aligned.update((a.id, b.id))
    overlap_rate = overlap_area / total_area if total_area else 0.0
    alignment_score = len(aligned) / len(foreground) if foreground else 0.0
    return overlap_rate, alignment_score


def _overflow_rate(state: LayoutState, conflicts: list[Conflict]) -> float:
    count = 0
    for element in state.elements:
        excursion = max(
            -element.x, -element.y,
            element.x + element.w - state.canvas.width,
            element.y + element.h - state.canvas.height,
        )
        if excursion > 0:
            count += 1
            conflicts.append(Conflict(
                conflict_type="overflow", element_ids=[element.id], severity="hard",
                detail=f"Farthest edge extends {excursion:g} pixels beyond the canvas.",
                numeric_value=excursion,
            ))
    return count / len(state.elements) if state.elements else 0.0


def _contrast_pass_rate(state: LayoutState, conflicts: list[Conflict]) -> float:
    passed = 0
    eligible = 0
    for index, element in enumerate(state.elements):
        if element.type not in TEXT_TYPES:
            continue
        eligible += 1
        fg = _luminance(element.fg_color or "#000000")
        bg = _luminance(effective_background(state, index))
        ratio = (max(fg, bg) + 0.05) / (min(fg, bg) + 0.05)
        threshold = 3.0 if (element.font_size or 0) >= 24 else 4.5
        if ratio >= threshold:
            passed += 1
        else:
            conflicts.append(Conflict(
                conflict_type="contrast", element_ids=[element.id], severity="hard",
                detail=f"Contrast {ratio} is below the required {threshold:.1f}.",
                numeric_value=ratio,
            ))
    return passed / eligible if eligible else 1.0


def evaluate_layout(state: LayoutState) -> dict:
    """Return OR, BOR, CPR, AS, CSR and the individual hard conflicts in input order per check."""
    conflicts: list[Conflict] = []
    overlap, alignment = _foreground_metrics(state.elements, conflicts)
    overflow = _overflow_rate(state, conflicts)
    contrast = _contrast_pass_rate(state, conflicts)
    metrics = {
        "OR": overlap,
        "BOR": overflow,
        "CPR": contrast,
        "AS": alignment,
        "CSR": float(overlap == 0 and overflow == 0 and contrast == 1),
    }
    return {"conflicts": conflicts, "metrics": metrics}
