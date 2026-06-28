"""Plain Python runtime adapter."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from ..protocol import CallContext, Tactic


class CallableTactic(Tactic[Any, Any]):
    """Wrap a normal Python callable as a tactic."""

    runtime_kind = "python"

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        input_type: Any = None,
        output_type: Any = None,
        package_ref: str | None = None,
        service_ref: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.fn = fn
        signature = inspect.signature(fn)
        inferred_input, inferred_output = _infer_types(signature)
        self.input_type = input_type if input_type is not None else inferred_input
        self.output_type = output_type if output_type is not None else inferred_output
        self._signature = signature
        self._is_async = inspect.iscoroutinefunction(fn)
        super().__init__(
            name=name or getattr(fn, "__name__", None) or "callable",
            description=(
                description if description is not None else inspect.getdoc(fn) or ""
            ),
            package_ref=package_ref,
            service_ref=service_ref,
            examples=examples,
            metadata=metadata,
        )

    def _run(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._is_async:
            return asyncio.run(self.fn(input_value, **self._call_kwargs(context, kwargs)))
        return self.fn(input_value, **self._call_kwargs(context, kwargs))

    async def _arun(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        call_kwargs = self._call_kwargs(context, kwargs)
        if self._is_async:
            return await self.fn(input_value, **call_kwargs)
        return await asyncio.to_thread(self.fn, input_value, **call_kwargs)

    def _call_kwargs(
        self,
        context: CallContext | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        call_kwargs = dict(kwargs)
        if context is not None and _accepts_kw(self._signature, "context"):
            call_kwargs.setdefault("context", context)
        return call_kwargs


def as_tactic(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    input_type: Any = None,
    output_type: Any = None,
    package_ref: str | None = None,
    service_ref: str | None = None,
    examples: list[dict[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CallableTactic:
    """Return a tactic wrapper for a Python callable."""

    return CallableTactic(
        fn,
        name=name,
        description=description,
        input_type=input_type,
        output_type=output_type,
        package_ref=package_ref,
        service_ref=service_ref,
        examples=examples,
        metadata=metadata,
    )


def _infer_types(signature: inspect.Signature) -> tuple[Any, Any]:
    input_type = None
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        } and parameter.name != "context":
            if parameter.annotation is not inspect.Parameter.empty:
                input_type = parameter.annotation
            break
    output_type = (
        signature.return_annotation
        if signature.return_annotation is not inspect.Signature.empty
        else None
    )
    return input_type, output_type


def _accepts_kw(signature: inspect.Signature, name: str) -> bool:
    if name in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
