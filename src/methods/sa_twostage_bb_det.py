"""Single-agent two-stage blackboard designer guided by deterministic detection.

Isolates sequential Round-0 decomposition (geometry first, then style) from
role-scoped field locks during revision: Round 0 uses two calls with the exact
shared GEOMETRY_RULES and STYLE_RULES, while correction rounds (t >= 1) invoke
one unlocked single agent over ALL_FIELDS receiving all hard conflicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..detector import evaluate_layout
from .arbiter import resolve_conflicts
from .blackboard import CRITIC, STYLE, _initial_geometry, _patched, messages
from .common import RunLog
from .rules import (
    GEOMETRY_RULES,
    OUTPUT_FORMAT_GEOMETRY_INIT,
    OUTPUT_FORMAT_STYLE_INIT_HEADER,
    STYLE_RULES,
    patch_format_instruction,
)
from .single_agent_bb_detector import ALL_FIELDS, CONFLICT_TYPES, DESIGNER, _edit_all_fields

if TYPE_CHECKING:
    from ..llm_client import LLMClient


def run_method(brief: dict, llm: LLMClient, max_rounds: int = 5) -> dict:
    if max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative.")
    log = RunLog(llm)

    # Round 0, stage 1: initialize geometry (verbatim Architect rules).
    geom_instruction = f"{OUTPUT_FORMAT_GEOMETRY_INIT} {GEOMETRY_RULES}"
    initial_geom = log.call(
        0,
        DESIGNER,
        messages(DESIGNER, geom_instruction, brief=brief),
        lambda obj: _initial_geometry(obj, brief, strict_keys=True),
    )
    state = _initial_geometry(initial_geom, brief, strict_keys=True)
    log.history[-1].update(state_version=state.version, init_stage="geometry")

    # Round 0, stage 2: initialize typography & color on the frozen geometry (verbatim Visual rules).
    all_ids = {e.id for e in state.elements}
    style_instruction = (
        f"{OUTPUT_FORMAT_STYLE_INIT_HEADER} {STYLE_RULES} "
        f"{patch_format_instruction(STYLE)}"
    )
    style_prompt = messages(
        DESIGNER,
        style_instruction,
        brief=brief,
        state=state.model_dump(),
        conflicts=[],
        allowed_ids=sorted(all_ids),
    )
    before_style = state
    style_reply = log.call(
        0,
        DESIGNER,
        style_prompt,
        lambda obj: _patched(
            obj, before_style, brief, all_ids, STYLE,
            strict_keys=True, require_initial_style=True,
        ),
    )
    state = _patched(
        style_reply, before_style, brief, all_ids, STYLE,
        strict_keys=True, require_initial_style=True,
    )
    log.history[-1].update(
        base_version=before_style.version,
        state_version=state.version,
        init_stage="style",
    )

    # Correction rounds (t = 1..max_rounds): unlocked single agent receives all hard conflicts.
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
        routed = [c for c in hard if c.conflict_type in CONFLICT_TYPES]
        if routed:
            state = _edit_all_fields(brief, state, routed, log, completed)
    return log.result(state, rounds_used=1 + completed)
