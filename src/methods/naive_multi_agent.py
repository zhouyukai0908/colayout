"""Three specialists sharing an append-only conversation and the latest complete proposal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from ..schema import LayoutState
from .common import RunLog, layout_prompt, validate_layout

if TYPE_CHECKING:
    from ..llm_client import LLMClient


ARCHITECT = "Layout Architect"
VISUAL = "Visual & Typography Agent"
CRITIC = "Compliance Critic"
_STACK_INDEX = TypeAdapter(int)


def _proposal(reply: dict, brief: dict) -> LayoutState:
    """Accept a bare LayoutState or the usual envelope and validate its inventory and copy."""
    payload = reply.get("layout", reply)
    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        raise ValueError("Provide the complete LayoutState with canvas and elements, bare or in layout.")
    elements = []
    for item in payload["elements"]:
        if isinstance(item, dict) and item.get("type") != "card" and "z_index" in item:
            if _STACK_INDEX.validate_python(item["z_index"]) <= 0:
                item = {**item, "z_index": 1}
        elements.append(item)
    return validate_layout({**payload, "elements": elements}, brief)


def _message(reply: dict, default: str) -> str:
    words = reply["message"].split() if isinstance(reply.get("message"), str) else []
    return " ".join(words[:120]) if words else default


def _keeps_layout(reply: dict) -> bool:
    unchanged = reply.get("action") == "KEEP_LAYOUT" or reply.get("status") == "KEEP_LAYOUT"
    return unchanged and "elements" not in reply and "layout" not in reply


def _completion_budget(brief: dict, role: str) -> int:
    items = brief["required_elements"]
    if role == CRITIC:
        return min(384, 192 + 12 * len(items))
    # Editors return each box and the required copy. Leave room for JSON keys and styling.
    copy_bytes = sum(len(item["text_or_desc"].encode("utf-8")) for item in items)
    return min(4096, 384 + 128 * len(items) + copy_bytes // 2)


def run_method(brief: dict, llm: LLMClient, max_rounds: int = 5) -> dict:
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    log = RunLog(llm)
    chat = [{"role": "user", "content": layout_prompt(brief)}]
    state: LayoutState | None = None
    turns = (
        (ARCHITECT, "Design the geometry and address any outstanding critique."),
        (VISUAL, "Review the latest proposal for typography, legibility, and colors."),
        (CRITIC, "Inspect the latest proposal for brief fidelity, overlap, overflow, "
                 "contrast, and alignment. Request revision only for a specific remaining "
                 "problem with an actionable correction. Accept satisfactory layouts."),
    )

    def validate_review(reply: dict) -> None:
        if reply.get("status") not in ("NO_CONFLICTS", "REVISE"):
            raise ValueError("The critic must set status to NO_CONFLICTS or REVISE.")

    def validate_editor(reply: dict) -> None:
        if _keeps_layout(reply):
            if state is None:
                raise ValueError("The first editor must provide a complete layout.")
        else:
            _proposal(reply, brief)

    for round_no in range(1, max_rounds + 1):
        for role, instruction in turns:
            response_format = (
                "Return JSON with status NO_CONFLICTS or REVISE and one short natural-language message."
                if role == CRITIC else
                "Return compact JSON with the complete LayoutState in layout and one short "
                "natural-language message to your teammates."
            )
            if state is not None and role != CRITIC:
                response_format += ' If unchanged, send {"action":"KEEP_LAYOUT","message":"..."} instead.'
            system = {"role": "system", "content": (
                f"You are the {role} in round {round_no}. {instruction} {response_format} "
                "Use one concise sentence (at most 120 words). Avoid reciting the layout."
            )}
            messages = [system, *chat]
            # Attach the document for this call only. The conversation retains every role's message.
            if state is not None:
                document = state.model_dump_json(exclude_unset=True, exclude_none=True)
                messages.append({"role": "user", "content": "Latest complete proposal:\n" + document})
            reply = log.call(
                round_no, role, messages, validate_review if role == CRITIC else validate_editor,
                max_tokens=_completion_budget(brief, role),
            )
            if role == CRITIC:
                message = _message(reply, "Review complete.")
                accepted = {"message": message, "status": reply["status"]}
                statement = f"{role}: {message}\nVerdict: {reply['status']}"
            elif _keeps_layout(reply):
                message = _message(reply, "Keep the current proposal.")
                accepted = {"message": message, "action": "KEEP_LAYOUT"}
                statement = f"{role} (unchanged): {message}"
            else:
                state = _proposal(reply, brief)
                proposal = state.model_dump(mode="json", exclude_unset=True)
                message = _message(reply, "See the proposed layout.")
                accepted = {"message": message, "layout": proposal}
                statement = f"{role} (proposed layout): {message}"
            log.history[-1]["response"] = accepted
            chat.append({"role": "assistant", "content": statement})
        if reply["status"] == "NO_CONFLICTS":
            break
    return log.result(state, round_no)
