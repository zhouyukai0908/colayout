"""Single-agent blackboard designer guided by the deterministic detector and priority arbiter.

Ablation isolating blackboard mechanics (versioned state, programmatic copy injection,
and element-keyed partial patches) from two-role specialization and disjoint field
write permissions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..detector import evaluate_layout
from ..schema import Conflict, Element, LayoutState
from .arbiter import resolve_conflicts
from .blackboard import CRITIC, GEOMETRY, STYLE, _patched, _updates, messages
from .common import RunLog, TEXT_ROLES, validate_layout

if TYPE_CHECKING:
    from ..llm_client import LLMClient


DESIGNER = "Layout Designer"
ALL_FIELDS = GEOMETRY | STYLE
CONFLICT_TYPES = {"overflow", "overlap", "contrast", "min_font"}


def _initial_all_fields(obj: dict, brief: dict) -> LayoutState:
    """Build version-0 snapshot from a combined geometry+style update list with programmatic copy injection."""
    required = brief["required_elements"]
    updates = _updates(obj, {e["id"] for e in required}, ALL_FIELDS)
    if required and not obj["updates"]:
        raise ValueError("Initial layout needs an update with an element id.")
    elements = []
    for item in required:
        name = item["id"]
        # Identical programmatic seed to blackboard._initial_geometry: brief canvas,
        # exact ID/type inventory, verbatim copy, and valid defaults overlaid with ALL_FIELDS.
        seed = {
            "id": name,
            "type": item["type"],
            "text": item["text_or_desc"] if item["type"] in TEXT_ROLES else None,
            "font_size": 16 if item["type"] in TEXT_ROLES else None,
            "x": 0,
            "y": 0,
            "w": brief["canvas_width"],
            "h": brief["canvas_height"],
            "z_index": 0 if item["type"] == "card" else 1,
        }
        elements.append(Element.model_validate({**seed, **updates.get(name, {})}, strict=True))
    canvas = {
        "width": brief["canvas_width"],
        "height": brief["canvas_height"],
        "bg_color": brief["canvas_bg_color"],
    }
    return validate_layout({"canvas": canvas, "elements": elements, "version": 0}, brief)


def _edit_all_fields(
    brief: dict, state: LayoutState, issues: list[Conflict], log: RunLog, round_no: int
) -> LayoutState:
    """Single unlocked designer updates any geometry/style field on the union of conflict element IDs."""
    ids = {i for c in issues for i in c.element_ids}
    instruction = (
        "Resolve the routed conflicts with targeted partial updates to the affected IDs. "
        "Preserve the design notes, alignment, and non-conflicting layout. "
        "Keep type 'card' at z_index 0 and all other elements (including 'image') at z_index >= 1. "
        "Use explicit positive font_size and hex fg_color for text types, "
        "optional hex bg_color (or null), and respect WCAG contrast and "
        "the brief's color constraints. "
        'Use {"updates":[{"id":"...","field":value}]}. Include only id '
        f"and the needed fields from: {', '.join(sorted(ALL_FIELDS))}. Use pixels."
    )
    prompt = messages(
        DESIGNER,
        instruction,
        brief=brief,
        state=state.model_dump(),
        conflicts=[c.model_dump() for c in issues],
        allowed_ids=sorted(ids),
    )

    def validate(obj: dict) -> LayoutState:
        return _patched(obj, state, brief, ids, ALL_FIELDS)

    response = log.call(round_no, DESIGNER, prompt, validate)
    after = validate(response)
    log.history[-1].update(base_version=state.version, state_version=after.version)
    return after


def run_method(brief: dict, llm: LLMClient, max_rounds: int = 5) -> dict:
    if max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative.")
    log = RunLog(llm)
    init_instruction = (
        "Initialize geometry, typography, and color for every required element. "
        'Return {"updates":[{"id":"...","x":0,"y":0,"w":10,"h":10,"z_index":1,'
        '"font_size":16,"fg_color":"#000000","bg_color":null}]} with one row per required element '
        f"and only needed fields from: {', '.join(sorted(ALL_FIELDS))}. "
        "Use positive pixel sizes. Use z_index 0 only for type 'card' and z_index >= 1 for all "
        "other foreground elements (including 'image' and text types). "
        "Plan for disjoint foreground bounds, edge alignment, readable text, and the specified "
        "header and footer zones. Use explicit positive font_size and hex fg_color for text types "
        "(title, subtitle, body, button, tag; omit or null for card/image), optional hex bg_color "
        "(or null), and respect WCAG contrast and the brief's color constraints."
    )
    initial = log.call(
        0,
        DESIGNER,
        messages(DESIGNER, init_instruction, brief=brief),
        lambda obj: _initial_all_fields(obj, brief),
    )
    state = _initial_all_fields(initial, brief)
    log.history[-1]["state_version"] = state.version

    completed = 0
    while True:
        conflicts = evaluate_layout(state)["conflicts"]
        log.history.append({
            "round": completed,
            "role": CRITIC,
            "state_version": state.version,
            "conflicts": [c.model_dump() for c in conflicts],
        })
        hard = [c for c in conflicts if c.severity == "hard"]
        if not hard:
            break
        if completed == max_rounds:
            before = state
            state = resolve_conflicts(state, None)
            log.history.append({
                "round": completed,
                "role": "Priority Arbiter",
                "base_version": before.version,
                "state_version": state.version,
                "changes": [b.model_dump() for a, b in zip(before.elements, state.elements) if a != b],
            })
            break
        completed += 1
        # Exact union of Architect ({overflow, overlap}) and Visual ({contrast, min_font}) hard conflicts.
        routed = [c for c in hard if c.conflict_type in CONFLICT_TYPES]
        if routed:
            state = _edit_all_fields(brief, state, routed, log, completed)
    return log.result(state, rounds_used=1 + completed)
