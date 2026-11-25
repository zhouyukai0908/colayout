"""Deterministic fallback for ordered overflow, overlap and contrast repairs."""

from itertools import combinations
from math import nextafter

from ..detector import _luminance, effective_background, evaluate_layout
from ..schema import Conflict, Element, LayoutState


def _bounded_length(start: float, end: float) -> float:
    """Subtract endpoints without rounding a reconstructed edge past its limit."""
    length = end - start
    if start + length > end:
        length = nextafter(length, 0.0)
    return length


def _clamp(element: Element, width: int, height: int) -> None:
    for coord, size, limit in (("x", "w", width), ("y", "h", height)):
        start, length = getattr(element, coord), getattr(element, size)
        if 0 <= start and start + length <= limit:
            continue
        length = min(length, float(limit))
        start = max(0.0, min(start, limit - length))
        setattr(element, coord, start)
        setattr(element, size, min(length, _bounded_length(start, limit)))


def _overlaps(a: Element, b: Element) -> bool:
    return (
        min(a.x + a.w, b.x + b.w) > max(a.x, b.x)
        and min(a.y + a.h, b.y + b.h) > max(a.y, b.y)
    )


def _separate(a: Element, b: Element, peers: list[Element], state: LayoutState) -> None:
    """Separate along the least-displacement axis. In ties preserve input order."""
    choices = []
    for coord, size, limit in (("x", "w", state.canvas.width), ("y", "h", state.canvas.height)):
        # Both orientations matter when one rectangle contains the other.
        for first, second in ((a, b), (b, a)):
            distance = getattr(first, coord) + getattr(first, size) - getattr(second, coord)
            choices.append((distance, coord, size, limit, first, second))
    _, coord, size, limit, first, second = min(choices, key=lambda choice: choice[0])
    first_start, second_start = getattr(first, coord), getattr(second, coord)
    first_end = first_start + getattr(first, size)
    second_end = second_start + getattr(second, size)

    # Prefer translation. A candidate must clear its partner and all other peers.
    for moving, position in ((second, first_end), (first, second_start - getattr(first, size))):
        if 0 <= position and position + getattr(moving, size) <= limit:
            candidate = moving.model_copy(update={coord: position})
            if all(not _overlaps(candidate, peer) for peer in peers if peer is not moving):
                setattr(moving, coord, position)
                return

    # A blocked pair can always share its intersecting strip. Cropping within
    # the original boxes creates no new intersections with a third element.
    low = max(first_start, second_start)
    high = min(first_end, second_end)
    cut = low + (high - low) / 2
    setattr(first, size, _bounded_length(first_start, cut))
    setattr(second, coord, cut)
    setattr(second, size, _bounded_length(cut, second_end))


def resolve_conflicts(state: LayoutState, reports: list[Conflict] | None = None) -> LayoutState:
    """Repair a copy, incrementing its version once when a field changes.

    The full framework rechecks geometry and contrast between priority stages.
    Supplying raw-JSON critic reports restricts every stage to the issues that
    critic actually reported, keeping detector feedback out of the ablation.
    """
    output = state.model_copy(deep=True)
    by_id = {e.id: e for e in output.elements}
    if reports is None:
        clamp_ids = list(by_id)
    else:
        clamp_ids = [i for c in reports if c.conflict_type == "overflow" for i in c.element_ids]
    for name in clamp_ids:
        _clamp(by_id[name], output.canvas.width, output.canvas.height)

    foreground = [e for e in output.elements if e.z_index >= 1 and e.type != "card"]
    if reports is None:
        pairs = combinations(foreground, 2)
    else:
        pairs = (
            (by_id[c.element_ids[0]], by_id[c.element_ids[1]])
            for c in reports if c.conflict_type == "overlap"
        )
    for a, b in pairs:
        if a in foreground and b in foreground and _overlaps(a, b):
            _separate(a, b, foreground, output)

    if reports is None:
        reports = evaluate_layout(output)["conflicts"]
    contrast_ids = {i for c in reports if c.conflict_type == "contrast" for i in c.element_ids}
    for index, element in enumerate(output.elements):
        if element.id in contrast_ids:
            bg = _luminance(effective_background(output, index))
            black_ratio = (bg + 0.05) / 0.05
            white_ratio = 1.05 / (bg + 0.05)
            element.fg_color = "#000000" if black_ratio >= white_ratio else "#FFFFFF"
    if output.elements != state.elements:
        output.version += 1
    return output
