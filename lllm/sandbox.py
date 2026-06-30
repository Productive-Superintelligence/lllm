"""Application-level sandbox utilities for tactics."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, Field

from .protocol import CallContext, Tactic, TacticError


class SandboxError(TacticError):
    """Base error raised by tactic sandbox wrappers."""


class SandboxLimitError(SandboxError):
    """Raised when a sandbox limit or allowlist check fails."""


class SandboxTimeoutError(SandboxError):
    """Raised when a sandboxed async tactic call exceeds its deadline."""


class SandboxPolicy(BaseModel):
    """Small call-boundary policy for a tactic wrapper.

    This is not OS-level isolation. It is a reusable protocol wrapper for
    service timeouts, payload-size budgets, and request-metadata allowlists.
    """

    timeout_seconds: float | None = Field(default=None, gt=0)
    max_input_bytes: int | None = Field(default=None, gt=0)
    max_output_bytes: int | None = Field(default=None, gt=0)
    allowed_metadata_keys: tuple[str, ...] | None = None


class SandboxedTactic(Tactic[Any, Any]):
    """Wrap another tactic with lightweight call-boundary checks."""

    runtime_kind = "sandbox"

    def __init__(
        self,
        tactic: Tactic[Any, Any],
        *,
        policy: SandboxPolicy | dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.tactic = tactic
        self.policy = (
            policy.model_copy(deep=True)
            if isinstance(policy, SandboxPolicy)
            else SandboxPolicy.model_validate({} if policy is None else policy)
        )
        info = tactic.info()
        self.input_type = tactic.input_type
        self.output_type = tactic.output_type
        super().__init__(
            name=name if name is not None else f"{tactic.tactic_name}_sandbox",
            description=(
                description
                if description is not None
                else f"Sandbox wrapper for {tactic.tactic_name}."
            ),
            package_ref=tactic.package_ref,
            service_ref=tactic.service_ref,
            examples=info.examples,
            metadata={
                "sandboxed_tactic": tactic.tactic_name,
                "sandboxed_runtime_kind": info.runtime_kind,
                "sandbox_policy": self.policy.model_dump(mode="json", exclude_none=True),
            },
        )

    def _run(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        context = context or CallContext()
        self._check_request(input_value, context)
        output = self.tactic.run(input_value, context=context, **kwargs)
        self._check_output(output)
        return output

    async def _arun(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        context = context or CallContext()
        self._check_request(input_value, context)
        call = self.tactic.arun(input_value, context=context, **kwargs)
        try:
            if self.policy.timeout_seconds is None:
                output = await call
            else:
                output = await asyncio.wait_for(call, timeout=self.policy.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise SandboxTimeoutError(
                f"Tactic exceeded timeout of {self.policy.timeout_seconds} seconds."
            ) from exc
        self._check_output(output)
        return output

    def _check_request(self, input_value: Any, context: CallContext) -> None:
        if self.policy.allowed_metadata_keys is not None:
            allowed = set(self.policy.allowed_metadata_keys)
            extra = sorted(set(context.metadata) - allowed)
            if extra:
                raise SandboxLimitError(
                    f"Context metadata keys are not allowed: {', '.join(extra)}"
                )
        if self.policy.max_input_bytes is not None:
            _check_size(
                "input",
                input_value,
                max_bytes=self.policy.max_input_bytes,
            )

    def _check_output(self, output_value: Any) -> None:
        if self.policy.max_output_bytes is not None:
            _check_size(
                "output",
                output_value,
                max_bytes=self.policy.max_output_bytes,
            )


def sandbox_tactic(
    tactic: Tactic[Any, Any],
    policy: SandboxPolicy | dict[str, Any] | None = None,
    **kwargs: Any,
) -> SandboxedTactic:
    """Return a sandbox wrapper for *tactic*."""

    return SandboxedTactic(tactic, policy=policy, **kwargs)


def _check_size(label: str, value: Any, *, max_bytes: int) -> None:
    size = len(_json_bytes(value))
    if size > max_bytes:
        raise SandboxLimitError(f"{label} is {size} bytes; limit is {max_bytes} bytes.")


def _json_bytes(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, default=str, sort_keys=True).encode("utf-8")


__all__ = [
    "SandboxError",
    "SandboxLimitError",
    "SandboxPolicy",
    "SandboxTimeoutError",
    "SandboxedTactic",
    "sandbox_tactic",
]
