"""Native runtime boundary.

The full native runtime is intentionally kept out of the protocol layer. This
minimal adapter preserves a clean tactic boundary for native-style objects while
the larger native runtime is ported intentionally under this package.
"""

from __future__ import annotations

import inspect
from typing import Any

from ...protocol import CallContext, Tactic
from .core import (
    APITypes,
    APIType,
    Dialog,
    DialogTreeNode,
    Function,
    FunctionCall,
    InvokeCost,
    Message,
    Modalities,
    Modality,
    Prompt,
    Role,
    Roles,
    StringFormatterRenderer,
    TokenLogprob,
    tool,
)


class NativeTacticAdapter(Tactic[Any, Any]):
    """Wrap a native-style object that exposes ``call``, ``run``, or ``arun``."""

    runtime_kind = "native"

    def __init__(
        self,
        native: Any,
        *,
        name: str | None = None,
        input_type: Any = None,
        output_type: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.native = native
        self.input_type = input_type or getattr(native, "input_type", None) or getattr(
            native,
            "input_model",
            None,
        )
        self.output_type = output_type or getattr(native, "output_type", None)
        super().__init__(
            name=name or getattr(native, "name", None) or type(native).__name__,
            description=inspect.getdoc(native) or "",
            metadata=metadata,
        )

    def _run(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        method = getattr(self.native, "run", None) or getattr(self.native, "call", None)
        if method is None:
            raise TypeError("Native object must define run() or call().")
        return _call_native(method, input_value, context=context, kwargs=kwargs)

    async def _arun(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        method = getattr(self.native, "arun", None)
        if method is None:
            return await super()._arun(input_value, context=context, **kwargs)
        result = _call_native(method, input_value, context=context, kwargs=kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def _call_native(
    method: Any,
    input_value: Any,
    *,
    context: CallContext | None,
    kwargs: dict[str, Any],
) -> Any:
    signature = inspect.signature(method)
    call_kwargs = dict(kwargs)
    if context is not None and "context" in signature.parameters:
        call_kwargs.setdefault("context", context)
    return method(input_value, **call_kwargs)


__all__ = [
    "APITypes",
    "APIType",
    "Dialog",
    "DialogTreeNode",
    "Function",
    "FunctionCall",
    "InvokeCost",
    "Message",
    "Modalities",
    "Modality",
    "NativeTacticAdapter",
    "Prompt",
    "Role",
    "Roles",
    "StringFormatterRenderer",
    "TokenLogprob",
    "tool",
]
