"""Inference hints: tell the serving stack what an agent is about to do.

An agent knows things an inference server cannot see from one request: which
calls belong to the same reasoning chain (a dialog), which chains were forked
from which (shared prompt prefixes), how urgent a call is, and roughly how
long the answer will be. Serving stacks with KV-cache-aware routing and
priority scheduling can use that to keep a chain's KV cache warm, route turns
to the worker that already holds the prefix, and schedule urgent work first.

:class:`InferenceHints` turns that knowledge into request fields for a
specific stack:

``dynamo``
    NVIDIA Dynamo: ``x-dynamo-session-id`` / ``x-dynamo-parent-session-id``
    headers and ``nvext.agent_hints`` (``priority``, ``strict_priority``,
    ``osl``, ``speculative_prefill``).
``vllm``
    vLLM's OpenAI server: ``priority`` (needs ``--scheduling-policy
    priority``), ``cache_salt``, and session headers for session-aware
    routers.
``sglang``
    SGLang's OpenAI server: ``priority`` (needs
    ``--enable-priority-scheduling``) and session headers for routers.
``openai``
    Any OpenAI-compatible endpoint: session headers only. Unknown headers are
    ignored by servers that do not use them, so this profile is always safe.

Hints are opt-in per agent. Session identity comes from the dialog being
answered: one dialog is one session, and a forked dialog names its parent,
so a server can tell that the child shares its parent's prefix.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

HintProfile = Literal["dynamo", "vllm", "sglang", "openai"]


class InferenceHints(BaseModel):
    """Serving hints an agent attaches to each model call.

    ``priority`` follows the Dynamo/SGLang convention: higher values are more
    urgent. The ``vllm`` profile converts it to vLLM's convention, where lower
    values are scheduled first.
    """

    model_config = ConfigDict(extra="forbid")

    profile: HintProfile = "openai"
    #: Send session identity derived from the dialog being answered.
    session: bool = True
    #: Optional prefix for session ids, e.g. a run id: ``"run-42:<dialog id>"``.
    session_prefix: Optional[str] = None
    priority: Optional[int] = None
    #: Dynamo router queue tier; higher tiers always go first.
    strict_priority: Optional[int] = Field(default=None, ge=0)
    #: Expected output length in tokens (Dynamo). Defaults to the agent's
    #: ``max_tokens`` / ``max_completion_tokens`` when those are set.
    osl: Optional[int] = Field(default=None, gt=0)
    #: Ask Dynamo to prefill the predicted next-turn prefix after this turn.
    speculative_prefill: Optional[bool] = None
    #: vLLM prefix-cache salt, to isolate cached prefixes between tenants.
    cache_salt: Optional[str] = None

    @field_validator("session_prefix", "cache_salt")
    @classmethod
    def _non_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    def session_ids(
        self,
        dialog_id: Optional[str],
        parent_dialog_id: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Session and parent-session ids for a dialog (or ``(None, None)``)."""

        if not self.session or not dialog_id:
            return None, None

        def named(value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            return f"{self.session_prefix}:{value}" if self.session_prefix else value

        return named(dialog_id), named(parent_dialog_id)

    def request_fields(
        self,
        *,
        dialog_id: Optional[str] = None,
        parent_dialog_id: Optional[str] = None,
        model_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return ``{"extra_headers": ..., "extra_body": ...}`` for one call.

        The result plugs into LiteLLM's ``completion(**kwargs)``; empty parts
        are omitted.
        """

        session_id, parent_id = self.session_ids(dialog_id, parent_dialog_id)
        headers: Dict[str, Any] = {}
        body: Dict[str, Any] = {}

        if self.profile == "dynamo":
            if session_id:
                headers["x-dynamo-session-id"] = session_id
            if parent_id:
                headers["x-dynamo-parent-session-id"] = parent_id
            agent_hints = {
                key: value
                for key, value in {
                    "priority": self.priority,
                    "strict_priority": self.strict_priority,
                    "osl": self.osl or _max_tokens(model_args),
                    "speculative_prefill": self.speculative_prefill,
                }.items()
                if value is not None
            }
            if agent_hints:
                body["nvext"] = {"agent_hints": agent_hints}
        else:
            if session_id:
                headers["x-session-id"] = session_id
            if parent_id:
                headers["x-parent-session-id"] = parent_id
            if self.profile == "vllm":
                if self.priority is not None:
                    body["priority"] = -self.priority
                if self.cache_salt is not None:
                    body["cache_salt"] = self.cache_salt
            elif self.profile == "sglang":
                if self.priority is not None:
                    body["priority"] = self.priority

        fields: Dict[str, Dict[str, Any]] = {}
        if headers:
            fields["extra_headers"] = headers
        if body:
            fields["extra_body"] = body
        return fields

    def final_request_fields(self, dialog_id: str) -> Dict[str, Dict[str, Any]]:
        """Fields for Dynamo's end-of-session request (``x-dynamo-session-final``).

        Send them on a dedicated minimal request when a dialog is finished so
        lifecycle-aware routers can release per-session state immediately.
        """

        fields = self.request_fields(dialog_id=dialog_id)
        if self.profile == "dynamo":
            fields.setdefault("extra_headers", {})["x-dynamo-session-final"] = "true"
        return fields


def merge_request_fields(
    model_args: Dict[str, Any],
    fields: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge hint fields into model args without overriding explicit values."""

    merged = dict(model_args)
    for key, values in fields.items():
        existing = merged.get(key)
        if isinstance(existing, dict):
            merged[key] = _deep_fill(dict(existing), values)
        elif existing is None:
            merged[key] = _deep_copy(values)
    return merged


def coerce_hints(value: Any) -> Optional[InferenceHints]:
    """Build hints from a config value: ``None``, a dict, or an instance."""

    if value is None or isinstance(value, InferenceHints):
        return value
    if isinstance(value, dict):
        return InferenceHints.model_validate(value)
    raise TypeError("inference_hints must be a mapping or InferenceHints.")


def _max_tokens(model_args: Optional[Dict[str, Any]]) -> Optional[int]:
    if not model_args:
        return None
    for key in ("max_tokens", "max_completion_tokens"):
        value = model_args.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _deep_fill(target: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in values.items():
        if key not in target:
            target[key] = _deep_copy(value)
        elif isinstance(target[key], dict) and isinstance(value, dict):
            target[key] = _deep_fill(dict(target[key]), value)
    return target


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    return value
