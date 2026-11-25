"""Single-designer revision using its own assessment of the current design."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..schema import LayoutState
from .common import RunLog, initial_layout, layout_prompt, require_text, validate_layout

if TYPE_CHECKING:
    from ..llm_client import LLMClient


def run_method(brief: dict, llm: LLMClient, max_rounds: int = 5) -> dict:
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    log = RunLog(llm)
    state = initial_layout(brief, log)
    rounds = 1

    def validate_revision(reply: dict) -> None:
        if reply.get("status") == "NO_CONFLICTS":
            return
        if reply.get("status") != "REVISED":
            raise ValueError("Set status to NO_CONFLICTS or REVISED.")
        require_text(reply, "critique")
        validate_layout(reply.get("layout"), brief)

    for rounds in range(2, max_rounds + 1):
        messages = [
            {"role": "system", "content": (
                "Review your own layout for brief fidelity, overlap, overflow, contrast, "
                "and alignment. If satisfied, return {\"status\":\"NO_CONFLICTS\"}. "
                "Otherwise return JSON with status REVISED, a short critique describing "
                "the observed issue, and layout containing your complete revised LayoutState."
            )},
            {"role": "user", "content": (
                layout_prompt(brief) + "\nYour current layout:\n" + state.model_dump_json()
            )},
        ]
        reply = log.call(rounds, "Layout Designer", messages, validate_revision)
        if reply["status"] == "NO_CONFLICTS":
            break
        state = LayoutState.model_validate(reply["layout"])
    return log.result(state, rounds)
