"""CoLayout with a deterministic compliance critic and priority fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..detector import evaluate_layout
from ..schema import Conflict, LayoutState
from .blackboard import CRITIC, run_blackboard
from .common import RunLog

if TYPE_CHECKING:
    from ..llm_client import LLMClient


def _inspect(brief: dict, state: LayoutState, log: RunLog, round_no: int) -> list[Conflict]:
    conflicts = evaluate_layout(state)["conflicts"]
    log.history.append({
        "round": round_no, "role": CRITIC, "state_version": state.version,
        "conflicts": [c.model_dump() for c in conflicts],
    })
    return conflicts


def run_method(brief: dict, llm: LLMClient, max_rounds: int = 5) -> dict:
    return run_blackboard(brief, llm, max_rounds, _inspect, detector_feedback=True)
