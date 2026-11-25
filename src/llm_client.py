"""OpenAI-compatible JSON completions with per-request usage accounting."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from openai import OpenAI


def model_name() -> str:
    return os.getenv("MODEL_NAME") or "gpt-4o-mini"


class InvalidCompletion(ValueError):
    """Validation exhausted its retry budget. Usage includes every attempt."""

    def __init__(self, reason: str, token_usage: dict[str, int]):
        super().__init__(f"Invalid JSON completion after 3 attempts: {reason}")
        self.token_usage = token_usage


class LLMClient:
    def __init__(self, *, client: OpenAI | None = None):
        self.model = model_name()
        self._cumulative_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.unreported_requests = 0
        if client is None:
            from dotenv import dotenv_values
            from openai import OpenAI

            base_url = os.getenv("OPENAI_BASE_URL") or None
            key = os.getenv("OPENAI_API_KEY")
            if not key and (base_url or "").rstrip("/") in ("", "https://api.openai.com/v1"):
                local_env = Path(__file__).resolve().parents[1] / ".env.local"
                key = dotenv_values(local_env, interpolate=False).get("OPENAI_API_KEY")
            if not key:
                raise ValueError("Set OPENAI_API_KEY for the selected endpoint.")
            client = OpenAI(api_key=key, base_url=base_url, timeout=120, max_retries=2)
        self.client = client

    @property
    def cumulative_usage(self) -> dict[str, int]:
        """Snapshot of provider-reported counts across successful and failed calls."""
        return self._cumulative_usage.copy()

    def complete_json(
        self,
        messages: list[dict],
        validator: Callable[[dict], object] | type[BaseModel] | None = None,
        *,
        max_tokens: int | None = None,
    ) -> tuple[dict, dict[str, int]]:
        """Get one validated object, counting tokens from rejected replies too."""
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
            raise ValueError("max_tokens must be a positive integer.")
        options = {} if max_tokens is None else {"max_tokens": max_tokens}
        conversation = list(messages)
        if not any("json" in str(message.get("content", "")).lower() for message in messages):
            conversation.insert(0, {"role": "system", "content": "Return a JSON object."})
        usage = dict.fromkeys(self._cumulative_usage, 0)
        for attempt in range(3):
            try:
                reply = self.client.chat.completions.create(
                    model=self.model, messages=conversation, response_format={"type": "json_object"},
                    **options,
                )
            except (Exception, KeyboardInterrupt):
                # A transport error can occur after the provider processed the request.
                self.unreported_requests += 1
                raise
            # Record every reported field before inspecting choices or validating JSON.
            reported = getattr(reply, "usage", None)
            missing = []
            for name in usage:
                count = getattr(reported, name, None)
                if type(count) is int and count >= 0:
                    usage[name] += count
                    self._cumulative_usage[name] += count
                else:
                    missing.append(name)
            if missing:
                self.unreported_requests += 1
                raise RuntimeError(f"The endpoint did not report valid {', '.join(missing)}.")
            message = reply.choices[0].message
            if getattr(message, "refusal", None):
                raise RuntimeError("The model declined the request.")
            content = message.content or ""
            try:
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise ValueError("The top-level JSON value must be an object.")
                if isinstance(validator, type) and issubclass(validator, BaseModel):
                    validator.model_validate(data)
                elif validator is not None:
                    validator(data)
                return data, usage
            except ValueError as exc:
                if attempt == 2:
                    raise InvalidCompletion(str(exc), usage) from exc
                conversation = [*conversation, {"role": "assistant", "content": content}, {
                    "role": "user", "content": f"Correct the validation error and return JSON.\n{exc}",
                }]
        raise AssertionError("unreachable")
