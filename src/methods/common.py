"""Shared prompt, validation, and measurement conventions for the benchmark."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..detector import evaluate_layout
from ..schema import Canvas, LayoutState

if TYPE_CHECKING:
    from ..llm_client import LLMClient


TEXT_ROLES = {"title", "subtitle", "body", "button", "tag"}


def layout_prompt(brief: dict) -> str:
    return (
        "Design the complete layout described in the brief below. The LayoutState JSON "
        "object has fields canvas {width, height, bg_color}, elements, and version "
        "(initially 0). Use the exact canvas and one element per required id and type. "
        "Each element has id, type, x, y, w, h (top-left coordinates and positive "
        "sizes in pixels), and z_index (0 for cards, at least 1 for foreground). "
        "Text types (title, subtitle, body, button, tag) have text copied verbatim from "
        "text_or_desc, positive font_size in px, and fg_color. Other types use text_or_desc "
        "as a design instruction. Use #RRGGBB colors and optional bg_color. "
        "Respect design_notes, keep foreground elements disjoint and inside the canvas, "
        "align related edges or centers, and ensure legible WCAG contrast.\n"
        + json.dumps(brief, ensure_ascii=False)
    )


def validate_layout(payload: dict, brief: dict) -> LayoutState:
    """Validate the snapshot and preserve the brief's canvas, inventory, and copy."""
    state = LayoutState.model_validate(payload)
    canvas = Canvas(width=brief["canvas_width"], height=brief["canvas_height"],
                    bg_color=brief["canvas_bg_color"])
    if (state.canvas.width != canvas.width or state.canvas.height != canvas.height
            or state.canvas.bg_color.lower() != canvas.bg_color.lower()):
        raise ValueError("Use the exact canvas dimensions and color from the brief.")
    required = {row["id"]: row for row in brief["required_elements"]}
    if {e.id for e in state.elements} != set(required):
        raise ValueError(f"Use exactly the required element IDs: {list(required)}")
    for e in state.elements:
        row = required[e.id]
        if e.type != row["type"]:
            raise ValueError(f"{e.id} must have type {row['type']}.")
        if e.w <= 0 or e.h <= 0:
            raise ValueError(f"{e.id} needs positive width and height.")
        if (e.type == "card") != (e.z_index == 0):
            raise ValueError(
                f"{e.id} (type='{e.type}') must use z_index 0 only for type 'card' and >= 1 for foreground."
            )
        if e.type in TEXT_ROLES and e.text != row["text_or_desc"]:
            raise ValueError(
                f"{e.id} must copy its required text_or_desc verbatim: "
                f"expected {row['text_or_desc']!r}, got {e.text!r}."
            )
    return state


class RunLog:
    """Collect validated agent replies, account for tokens, and score the final state."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.history: list[dict] = []
        self.total_tokens = 0

    def call(
        self,
        round_no: int,
        role: str,
        messages: list[dict],
        validator: Callable[[dict], object] | None = None,
        *,
        max_tokens: int | None = None,
    ) -> dict:
        options = {"max_tokens": max_tokens} if max_tokens is not None else {}
        try:
            response, usage = self.llm.complete_json(messages, validator=validator, **options)
        except Exception as exc:
            exc.colayout_round = round_no
            exc.colayout_role = role
            raise
        self.total_tokens += usage["total_tokens"]
        entry = {
            "round": round_no, "role": role, "response": response, "token_usage": usage,
        }
        last_meta = getattr(self.llm, "last_call_metadata", None)
        if isinstance(last_meta, dict):
            if last_meta.get("resolved_model"):
                entry["resolved_model"] = last_meta["resolved_model"]
            if last_meta.get("system_fingerprint"):
                entry["system_fingerprint"] = last_meta["system_fingerprint"]
            if "format_rejections" in last_meta:
                entry["format_rejections"] = last_meta["format_rejections"]
            if last_meta.get("rejection_reasons"):
                entry["rejection_reasons"] = last_meta["rejection_reasons"]
        self.history.append(entry)
        return response

    def result(self, state: LayoutState, rounds_used: int) -> dict:
        report = evaluate_layout(state)
        return {
            "final_state": state.model_dump(mode="json"),
            "metrics": report["metrics"],
            "conflicts": [c.model_dump(mode="json") for c in report["conflicts"]],
            "rounds_used": rounds_used,
            "total_tokens": self.total_tokens,
            "history": self.history,
        }


def initial_layout(brief: dict, log: RunLog) -> LayoutState:
    messages = [
        {"role": "system", "content": (
            "You are a layout designer. Plan the placement, hierarchy, typography, "
            "and constraint checks internally. Return only the final LayoutState JSON object."
        )},
        {"role": "user", "content": layout_prompt(brief)},
    ]
    response = log.call(1, "Layout Designer", messages, lambda obj: validate_layout(obj, brief))
    return LayoutState.model_validate(response)


def require_text(response: dict, field: str) -> None:
    if not isinstance(response.get(field), str) or not response[field].strip():
        raise ValueError(f"Include a nonempty {field} string.")
