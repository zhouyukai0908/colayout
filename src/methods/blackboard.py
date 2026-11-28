"""Versioned snapshots and role-scoped JSON patches for the two CoLayout variants."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..schema import Conflict, Element, LayoutState
from .arbiter import resolve_conflicts
from .common import RunLog, TEXT_ROLES, validate_layout
from .rules import (
    GEOMETRY_RULES,
    OUTPUT_FORMAT_GEOMETRY_INIT,
    OUTPUT_FORMAT_STYLE_INIT_HEADER,
    REVISION_RULES,
    STYLE_RULES,
    patch_format_instruction,
)

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


def _updates(
    obj: dict,
    allowed: set[str],
    fields: set[str],
    *,
    strict_keys: bool = True,
) -> dict:
    """Validate and collect patch rows for routed IDs and the role's writable fields.

    In live runs (strict_keys=True), never silently drop model output: reject
    unexpected top-level keys, nested wrapper dicts (such as literal "field"),
    unknown/read-only fields, disallowed element IDs, and empty rows.
    Legacy replay of pre-v2 logs may pass strict_keys=False.
    """
    if strict_keys:
        extra_top = set(obj) - {"updates"}
        if extra_top:
            raise ValueError(
                f"Unexpected top-level key(s) {sorted(extra_top)}; return only an object with 'updates'."
            )
    rows = obj.get("updates")
    if not isinstance(rows, list):
        raise ValueError("Return an object with an updates array.")
    updates: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each update must be an object with an id.")
        name = row.get("id")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each update needs a nonempty string id.")
        if strict_keys:
            if "field" in row:
                raise ValueError(
                    f"Update row for {name!r} uses nested key 'field'; "
                    f"put fields directly on the update object from: {', '.join(sorted(fields))}."
                )
            for key, value in row.items():
                if isinstance(value, dict):
                    raise ValueError(
                        f"Update row for {name!r} has nested object under {key!r}; "
                        f"put scalar fields directly on the update object."
                    )
            if name not in allowed:
                raise ValueError(
                    f"Element id {name!r} is not in allowed_ids {sorted(allowed)}."
                )
            unknown = set(row) - fields - {"id"}
            if unknown:
                raise ValueError(
                    f"Update row for {name!r} includes unknown or non-writable field(s) "
                    f"{sorted(unknown)}; allowed fields are: {', '.join(sorted(fields))}."
                )
            patch = {k: v for k, v in row.items() if k in fields}
            if not patch:
                raise ValueError(
                    f"Update row for {name!r} must include at least one field from: "
                    f"{', '.join(sorted(fields))}."
                )
            updates.setdefault(name, {}).update(patch)
        elif name in allowed:
            patch = {k: v for k, v in row.items() if k in fields}
            if patch:
                updates.setdefault(name, {}).update(patch)
    return updates


def _validate_style_completeness(updates: dict[str, dict], brief: dict) -> None:
    """Require explicit fg_color/font_size for all text elements and non-null bg_color for all cards."""
    missing_fg = []
    missing_fs = []
    missing_card_bg = []
    for item in brief["required_elements"]:
        name, etype = item["id"], item["type"]
        patch = updates.get(name, {})
        if etype in TEXT_ROLES:
            if patch.get("fg_color") is None:
                missing_fg.append(name)
            if patch.get("font_size") is None:
                missing_fs.append(name)
        elif etype == "card":
            if patch.get("bg_color") is None:
                missing_card_bg.append(name)
    errors = []
    if missing_fg:
        errors.append(f"missing fg_color for text element(s) {missing_fg}")
    if missing_fs:
        errors.append(f"missing font_size for text element(s) {missing_fs}")
    if missing_card_bg:
        errors.append(f"missing non-null bg_color for card element(s) {missing_card_bg}")
    if errors:
        raise ValueError("Incomplete initial styling: " + "; ".join(errors) + ".")


def _initial_geometry(obj: dict, brief: dict, *, strict_keys: bool = True) -> LayoutState:
    required = brief["required_elements"]
    updates = _updates(obj, {e["id"] for e in required}, GEOMETRY, strict_keys=strict_keys)
    if required and not obj["updates"]:
        raise ValueError("Initial geometry needs an update with an element id.")
    if strict_keys:
        missing_geom = [
            item["id"]
            for item in required
            if not GEOMETRY.issubset(updates.get(item["id"], {}).keys())
        ]
        if missing_geom:
            raise ValueError(
                f"Initial geometry must provide all fields {sorted(GEOMETRY)} for every required element; "
                f"incomplete: {missing_geom}."
            )
    elements = []
    for item in required:
        name = item["id"]
        if strict_keys:
            # Do not pre-fill passing default colors or font sizes before style init.
            seed = {
                "id": name,
                "type": item["type"],
                "text": item["text_or_desc"] if item["type"] in TEXT_ROLES else None,
                "font_size": None,
                "fg_color": None,
                "bg_color": None,
            }
        else:
            # Legacy pre-v2 replay seed for historical results/raw/ logs.
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


def _patched(
    obj: dict,
    state: LayoutState,
    brief: dict,
    ids: set[str],
    fields: set[str],
    *,
    strict_keys: bool = True,
    require_initial_style: bool = False,
) -> LayoutState:
    updates = _updates(obj, ids, fields, strict_keys=strict_keys)
    if require_initial_style and strict_keys:
        _validate_style_completeness(updates, brief)
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
        task = f"{OUTPUT_FORMAT_STYLE_INIT_HEADER} {STYLE_RULES}"
    else:
        task = REVISION_RULES
    instruction = f"{task} {patch_format_instruction(fields)}"
    prompt = messages(role, instruction, brief=brief, state=state.model_dump(),
                      conflicts=[c.model_dump() for c in issues], allowed_ids=sorted(ids))

    def validate(obj: dict) -> LayoutState:
        return _patched(
            obj, state, brief, ids, fields,
            strict_keys=True,
            require_initial_style=initializing_style,
        )

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
    instruction = f"{OUTPUT_FORMAT_GEOMETRY_INIT} {GEOMETRY_RULES}"
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
