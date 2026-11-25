"""Versioned snapshots and role-scoped JSON patches for the two CoLayout variants."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..schema import Conflict, Element, LayoutState
from .arbiter import resolve_conflicts
from .common import RunLog, TEXT_ROLES, validate_layout

if TYPE_CHECKING:
    from ..llm_client import LLMClient


ARCHITECT = "Layout Architect"
VISUAL = "Visual & Typography Agent"
CRITIC = "Compliance Critic"
GEOMETRY = {"x", "y", "w", "h", "z_index"}
STYLE = {"font_size", "fg_color", "bg_color"}


def messages(role: str, instruction: str, **context) -> list[dict]:
    return [
        {"role": "system", "content": f"You are the {role}. {instruction} Return only JSON."},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]


def _updates(obj: dict, allowed: set[str], fields: set[str]) -> dict:
    """Project patch rows onto the routed IDs and the role's writable fields."""
    rows = obj.get("updates")
    if not isinstance(rows, list):
        raise ValueError("Return an object with an updates array.")
    updates = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each update must be an object with an id.")
        name = row.get("id")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each update needs a nonempty string id.")
        if name in allowed:
            patch = {k: v for k, v in row.items() if k in fields}
            if patch:
                updates.setdefault(name, {}).update(patch)
    return updates


def _initial_geometry(obj: dict, brief: dict) -> LayoutState:
    required = brief["required_elements"]
    updates = _updates(obj, {e["id"] for e in required}, GEOMETRY)
    if required and not obj["updates"]:
        raise ValueError("Initial geometry needs an update with an element id.")
    elements = []
    for item in required:
        name = item["id"]
        # At initialization there is no previous geometry to retain. Start each
        # missing box at the canvas bounds and preserve the brief's inventory.
        seed = {
            "id": name, "type": item["type"],
            "text": item["text_or_desc"] if item["type"] in TEXT_ROLES else None,
            "font_size": 16 if item["type"] in TEXT_ROLES else None,
            "x": 0, "y": 0, "w": brief["canvas_width"], "h": brief["canvas_height"],
            "z_index": 0 if item["type"] == "card" else 1,
        }
        elements.append(Element.model_validate({**seed, **updates.get(name, {})}, strict=True))
    canvas = {"width": brief["canvas_width"], "height": brief["canvas_height"],
              "bg_color": brief["canvas_bg_color"]}
    return validate_layout({"canvas": canvas, "elements": elements, "version": 0}, brief)


def _patched(obj: dict, state: LayoutState, brief: dict, ids: set[str],
             fields: set[str]) -> LayoutState:
    updates = _updates(obj, ids, fields)
    payload = state.model_dump()
    for index, element in enumerate(payload["elements"]):
        patch = updates.get(element["id"])
        if patch:
            payload["elements"][index] = Element.model_validate({**element, **patch}, strict=True)
    result = validate_layout(payload, brief)
    result.version += int(result.elements != state.elements)
    return result


def _edit(brief: dict, state: LayoutState, issues: list[Conflict], log: RunLog,
          round_no: int, role: str, initializing_style: bool = False) -> LayoutState:
    fields = GEOMETRY if role == ARCHITECT else STYLE
    ids = {i for c in issues for i in c.element_ids}
    if initializing_style:
        ids = {e.id for e in state.elements}
        task = ("Initialize typography and color for every ID. Use bg_color (hex or null) "
                "for containers and images, and explicit positive font_size and hex fg_color "
                "for all text types. Respect the brief's color constraints.")
    else:
        task = ("Resolve the routed conflicts with targeted partial updates to the affected IDs. "
                "Preserve the design notes, alignment, and non-conflicting layout. Keep cards "
                "at z_index 0 and all other elements at z_index >= 1.")
    instruction = (task + ' Use {"updates":[{"id":"...","field":value}]}. Include only id '
                   f"and the needed fields from: {', '.join(sorted(fields))}. Use pixels.")
    prompt = messages(role, instruction, brief=brief, state=state.model_dump(),
                      conflicts=[c.model_dump() for c in issues], allowed_ids=sorted(ids))

    def validate(obj: dict) -> LayoutState:
        return _patched(obj, state, brief, ids, fields)

    response = log.call(round_no, role, prompt, validate)
    after = validate(response)
    log.history[-1].update(base_version=state.version, state_version=after.version)
    return after


Critic = Callable[[dict, LayoutState, RunLog, int], list[Conflict]]


def run_blackboard(brief: dict, llm: LLMClient, max_rounds: int, critic: Critic,
                   *, detector_feedback: bool) -> dict:
    if max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative.")
    log = RunLog(llm)
    instruction = (
        'Initialize geometry. Return {"updates":[{"id":"...","x":0,"y":0,"w":10,'
        '"h":10,"z_index":1}]} with only those fields and one row per required element. Use positive pixel '
        "sizes. Use z_index 0 for cards and >=1 for foreground. Plan for disjoint foreground "
        "bounds, edge alignment, readable text and the specified header and footer zones."
    )
    initial = log.call(0, ARCHITECT, messages(ARCHITECT, instruction, brief=brief),
                       lambda obj: _initial_geometry(obj, brief))
    state = _initial_geometry(initial, brief)
    log.history[-1]["state_version"] = state.version
    state = _edit(brief, state, [], log, 0, VISUAL, initializing_style=True)
    completed = 0
    while True:
        # Inspect the latest version, including after the last allowed correction.
        issues = critic(brief, state, log, completed)
        hard = [c for c in issues if c.severity == "hard"]
        if not hard:
            break
        if completed == max_rounds:
            before = state
            state = resolve_conflicts(state, None if detector_feedback else hard)
            log.history.append({
                "round": completed, "role": "Priority Arbiter",
                "base_version": before.version, "state_version": state.version,
                "changes": [b.model_dump() for a, b in zip(before.elements, state.elements) if a != b],
            })
            break
        completed += 1
        for role, types in ((ARCHITECT, {"overflow", "overlap"}), (VISUAL, {"contrast", "min_font"})):
            routed = [c for c in hard if c.conflict_type in types]
            if routed:
                state = _edit(brief, state, routed, log, completed, role)
    return log.result(state, rounds_used=1 + completed)
