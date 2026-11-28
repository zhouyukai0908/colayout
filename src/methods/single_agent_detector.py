"""Single-agent layout designer guided by the deterministic detector and priority arbiter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ..detector import evaluate_layout
from ..schema import LayoutState
from .arbiter import resolve_conflicts
from .common import RunLog, validate_layout
from .rules import (
    GEOMETRY_RULES,
    OUTPUT_FORMAT_FULL_LAYOUT_INIT,
    OUTPUT_FORMAT_FULL_LAYOUT_REVISION,
    STYLE_RULES,
)

if TYPE_CHECKING:
    from ..llm_client import LLMClient


DESIGNER = "Layout Designer"
CRITIC = "Compliance Critic"


def _aligned_layout_prompt(brief: dict) -> str:
    return (
        f"{OUTPUT_FORMAT_FULL_LAYOUT_INIT} {GEOMETRY_RULES} {STYLE_RULES}\n"
        + json.dumps(brief, ensure_ascii=False)
    )


def _initial_state(brief: dict, log: RunLog) -> LayoutState:
    messages = [
        {"role": "system", "content": (
            "You are a layout designer. Plan the placement, hierarchy, typography, "
            "and constraint checks internally. Return only the final LayoutState JSON object."
        )},
        {"role": "user", "content": _aligned_layout_prompt(brief)},
    ]
    response = log.call(0, DESIGNER, messages, lambda obj: validate_layout(obj, brief))
    state = validate_layout(response, brief)
    state.version = 0
    log.history[-1]["state_version"] = state.version
    return state


def run_method(brief: dict, llm: LLMClient, max_rounds: int = 5) -> dict:
    if max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative.")
    log = RunLog(llm)
    state = _initial_state(brief, log)
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
        messages = [
            {"role": "system", "content": f"You are a layout designer. {OUTPUT_FORMAT_FULL_LAYOUT_REVISION}"},
            {"role": "user", "content": (
                _aligned_layout_prompt(brief)
                + "\nYour current layout:\n" + state.model_dump_json()
                + "\nHard conflicts reported by the deterministic detector:\n"
                + json.dumps([c.model_dump() for c in hard], ensure_ascii=False)
            )},
        ]
        response = log.call(completed, DESIGNER, messages, lambda obj: validate_layout(obj, brief))
        after = validate_layout(response, brief)
        after.version = state.version + int(after.elements != state.elements)
        log.history[-1].update(base_version=state.version, state_version=after.version)
        state = after
    return log.result(state, rounds_used=1 + completed)
