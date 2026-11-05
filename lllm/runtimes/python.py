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
        tactic_name = (
            name if name is not None else getattr(fn, "__name__", None) or "callable"
        )
        super().__init__(
            name=tactic_name,
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
    """Return a tactic wrapper for a Python callable.

    A tactic passed without overrides is returned as it is.
    """

    overrides = (name, description, input_type, output_type, package_ref, service_ref, examples, metadata)
    if isinstance(fn, Tactic) and all(value is None for value in overrides):
        return fn

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


def coerce_tactic(value: Any, *, name: str | None = None) -> Tactic[Any, Any]:
    """Turn an imported entrypoint value into a tactic.

    Accepts, in order: a ``Tactic`` subclass (instantiated) or instance; a
    zero-argument factory returning any of the accepted shapes; an object with
    a ``run`` method; or a plain callable, wrapped with :func:`as_tactic`.
    ``name`` names the tactic when one has to be created.
    """

    kwargs: dict[str, Any] = {} if name is None else {"name": name}
    if isinstance(value, type) and issubclass(value, Tactic):
        return value(**kwargs)
    if isinstance(value, Tactic):
        return value
    if callable(value) and _callable_without_args(value):
        produced = value()
        if isinstance(produced, type) and issubclass(produced, Tactic):
            return produced(**kwargs)
        if isinstance(produced, Tactic):
            return produced
        if callable(getattr(produced, "run", None)):
            return as_tactic(produced.run, **kwargs)
        if callable(produced):
            return as_tactic(produced, **kwargs)
    if callable(getattr(value, "run", None)):
        return as_tactic(value.run, **kwargs)
    if callable(value):
        return as_tactic(value, **kwargs)
    raise TypeError(f"Value does not resolve to a tactic or callable: {value!r}")


def _callable_without_args(value: Any) -> bool:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return False
    return not any(
        parameter.default is inspect.Parameter.empty
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        for parameter in signature.parameters.values()
    )
