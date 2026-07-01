"""Adapters between preserved native objects and the v2 tactic protocol."""

from __future__ import annotations

import inspect
from typing import Any

from ...protocol import CallContext, Tactic


class NativeTacticAdapter(Tactic[Any, Any]):
    """Wrap a native-style object behind the v2 ``Tactic`` boundary."""

    runtime_kind = "native"

    def __init__(
        self,
        native: Any,
        *,
        name: str | None = None,
        description: str | None = None,
        input_type: Any = None,
        output_type: Any = None,
        package_ref: str | None = None,
        service_ref: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.native = native
        self.input_type = input_type or getattr(native, "input_type", None) or getattr(
            native,
            "input_model",
            None,
        )
        self.output_type = output_type or getattr(native, "output_type", None) or getattr(
            native,
            "output_model",
            None,
        )
        tactic_name = (
            name
            if name is not None
            else getattr(native, "name", None) or type(native).__name__
        )
        super().__init__(
            name=tactic_name,
            description=(
                description
                if description is not None
                else inspect.getdoc(native) or inspect.getdoc(type(native)) or ""
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
        method = _select_sync_method(self.native)
        return _call_native(method, input_value, context=context, kwargs=kwargs)

    async def _arun(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        method = _select_async_method(self.native)
        if method is None:
            return await super()._arun(input_value, context=context, **kwargs)
        result = _call_native(method, input_value, context=context, kwargs=kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def capabilities(self) -> set[str]:
        supported = {"run", "arun"}
        native_capabilities = getattr(self.native, "capabilities", None)
        if callable(native_capabilities):
            for capability in native_capabilities():
                if capability in {"stream", "events"}:
                    supported.add(capability)
        elif _has_static_callable(self.native, "stream"):
            supported.add("stream")
        return supported


def _select_sync_method(native: Any) -> Any:
    method = getattr(native, "run", None)
    if method is not None:
        return method
    if _is_native_tactic(native):
        return native.__call__
    method = getattr(native, "call", None)
    if method is not None:
        return method
    raise TypeError("Native object must define run(), call(), or native Tactic.__call__().")


def _select_async_method(native: Any) -> Any | None:
    method = getattr(native, "arun", None)
    if method is not None:
        return method
    method = getattr(native, "acall", None)
    if method is not None:
        return method
    return None


def _call_native(
    method: Any,
    input_value: Any,
    *,
    context: CallContext | None,
    kwargs: dict[str, Any],
) -> Any:
    signature = inspect.signature(method)
    call_kwargs = dict(kwargs)
    if context is not None and _accepts_keyword(signature, "context"):
        call_kwargs.setdefault("context", context)
    return method(input_value, **call_kwargs)


def _is_native_tactic(value: Any) -> bool:
    try:
        from .core.tactic import Tactic as NativeTacticProtocol
    except Exception:
        return False
    return isinstance(value, NativeTacticProtocol)


def _has_static_callable(value: Any, name: str) -> bool:
    try:
        attribute = inspect.getattr_static(value, name)
    except AttributeError:
        return False
    descriptor_value = getattr(attribute, "__func__", attribute)
    return callable(descriptor_value)


def _accepts_keyword(signature: inspect.Signature, name: str) -> bool:
    if name in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


__all__ = ["NativeTacticAdapter"]
