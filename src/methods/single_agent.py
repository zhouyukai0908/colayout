"""Generate a complete layout in one round."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .common import RunLog, initial_layout

if TYPE_CHECKING:
    from ..llm_client import LLMClient


def run_method(brief: dict, llm: LLMClient, max_rounds: int = 5) -> dict:
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    log = RunLog(llm)
    state = initial_layout(brief, log)
    return log.result(state, 1)
