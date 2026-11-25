"""CoLayout ablation whose compliance critic inspects the raw blackboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..schema import Conflict, LayoutState
from .blackboard import CRITIC, messages, run_blackboard
from .common import RunLog

if TYPE_CHECKING:
    from ..llm_client import LLMClient


def _inspect(brief: dict, state: LayoutState, log: RunLog, round_no: int) -> list[Conflict]:
    prompt = messages(CRITIC, (
        "Inspect the raw state for foreground overlap, canvas overflow, and low WCAG text "
        "contrast. The contrast threshold is 4.5, or 3.0 for font_size >=24. Ignore background "
        "cards when inspecting overlap. Report hard issues only. Return an object like "
        '{"conflicts":[{"conflict_type":"overlap","element_ids":["a","b"],'
        '"detail":"description","numeric_value":12}]}. For overflow and contrast use one '
        "affected ID. Use numeric_value for intersection area, overflow pixels or contrast ratio. "
        'When satisfied, return {"conflicts":[]}.'
    ), brief=brief, state=state.model_dump())
    ids = {e.id for e in state.elements}

    def validate(obj: dict) -> list[Conflict]:
        rows = obj.get("conflicts")
        if not isinstance(rows, list):
            raise ValueError("Return a conflicts array.")
        issues = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each conflict must be an object.")
            issue = Conflict.model_validate({**row, "severity": "hard"})
            n = 2 if issue.conflict_type == "overlap" else 1
            names = set(issue.element_ids)
            if len(names) != n or len(issue.element_ids) != n or not names <= ids:
                raise ValueError(f"Supply {n} distinct existing element IDs for {issue.conflict_type}.")
            issues.append(issue)
        return issues

    response = log.call(round_no, CRITIC, prompt, validate)
    log.history[-1]["state_version"] = state.version
    return validate(response)


def run_method(brief: dict, llm: LLMClient, max_rounds: int = 5) -> dict:
    return run_blackboard(brief, llm, max_rounds, _inspect, detector_feedback=False)
