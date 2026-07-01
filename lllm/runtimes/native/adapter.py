"""Adapters between preserved native objects and the v2 tactic protocol."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from ...protocol import CallContext, Tactic
from ...protocol._validation import copy_boundary_value


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
        run_kwargs: Mapping[str, Any] | None = None,
        include_context_metadata: bool = True,
    ) -> None:
        self.native = native
        self.run_kwargs = _runtime_kwargs(run_kwargs)
        if not isinstance(include_context_metadata, bool):
            raise TypeError("include_context_metadata must be a boolean.")
        self.include_context_metadata = include_context_metadata
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
        return _call_native(
            self.native,
            method,
            input_value,
            context=context,
            run_kwargs=self.run_kwargs,
            call_kwargs=kwargs,
            include_context_metadata=self.include_context_metadata,
        )

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
        result = _call_native(
            self.native,
            method,
            input_value,
            context=context,
            run_kwargs=self.run_kwargs,
            call_kwargs=kwargs,
            include_context_metadata=self.include_context_metadata,
        )
        if inspect.isawaitable(result):
            return await result
        return result

    def stream(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ):
        method = getattr(self.native, "stream", None)
        if method is None:
            return super().stream(input_value, context=context, **kwargs)
        result = _call_native(
            self.native,
            method,
            input_value,
            context=context,
            run_kwargs=self.run_kwargs,
            call_kwargs=kwargs,
            include_context_metadata=self.include_context_metadata,
        )
        yield from _iter_native_stream(result)

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
    native: Any,
    method: Any,
    input_value: Any,
    *,
    context: CallContext | None,
    run_kwargs: Mapping[str, Any],
    call_kwargs: dict[str, Any],
    include_context_metadata: bool,
) -> Any:
    signature = inspect.signature(method)
    merged_kwargs = _runtime_kwargs(run_kwargs)
    merged_kwargs.update(_runtime_kwargs(call_kwargs))
    if context is not None and _accepts_context(native, signature):
        merged_kwargs.setdefault("context", context)
    if (
        include_context_metadata
        and context is not None
        and "metadata" not in merged_kwargs
        and _accepts_keyword(signature, "metadata")
    ):
        merged_kwargs["metadata"] = _context_metadata(context)
    return method(input_value, **merged_kwargs)


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


def _accepts_context(native: Any, signature: inspect.Signature) -> bool:
    if _accepts_explicit_keyword(signature, "context"):
        return True
    if _is_native_tactic(native):
        call = getattr(native, "call", None)
        if call is None:
            return False
        try:
            return _accepts_keyword(inspect.signature(call), "context")
        except (TypeError, ValueError):
            return False
    return _accepts_keyword(signature, "context")


def _accepts_explicit_keyword(signature: inspect.Signature, name: str) -> bool:
    parameter = signature.parameters.get(name)
    if parameter is None:
        return False
    return parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


def _accepts_keyword(signature: inspect.Signature, name: str) -> bool:
    if name in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _runtime_kwargs(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("run_kwargs must be a mapping.")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("run_kwargs keys must be strings.")
        copied[key] = copy_boundary_value(item)
    return copied


def _context_metadata(context: CallContext) -> dict[str, Any]:
    metadata = copy_boundary_value(context.metadata)
    metadata.setdefault("lllm_request_id", context.request_id)
    if context.trace_id is not None:
        metadata.setdefault("lllm_trace_id", context.trace_id)
    if context.span_id is not None:
        metadata.setdefault("lllm_span_id", context.span_id)
    if context.package_ref is not None:
        metadata.setdefault("lllm_package_ref", context.package_ref)
    if context.service_ref is not None:
        metadata.setdefault("lllm_service_ref", context.service_ref)
    if context.tactic_ref is not None:
        metadata.setdefault("lllm_tactic_ref", context.tactic_ref)
    if context.endpoint is not None:
        metadata.setdefault("lllm_endpoint", context.endpoint)
    if context.tags:
        metadata.setdefault("lllm_tags", dict(context.tags))
    return metadata


def _iter_native_stream(result: Any):
    if hasattr(result, "__iter__") and not isinstance(
        result,
        (str, bytes, bytearray, dict),
    ):
        yield from result
        return
    yield result


__all__ = ["NativeTacticAdapter"]
