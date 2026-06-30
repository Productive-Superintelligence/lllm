"""Pydantic AI runtime adapter.

LLLM delegates agent behavior to the user-provided Pydantic AI object. The
adapter only supplies the typed tactic boundary and service-friendly metadata.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

from ..protocol import CallContext, Tactic, TacticInfo
from ..protocol._validation import copy_boundary_value

InputMode = Literal["auto", "json", "dict", "python", "text"]
ResultMode = Literal["output", "result"]
StreamMode = Literal["output", "text", "response", "raw"]


def _copy_runtime_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_runtime_value(item) for item in value)
    if isinstance(value, set):
        return {_copy_runtime_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_copy_runtime_value(item) for item in value)
    return value


def _copy_runtime_kwargs(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("run_kwargs must be a mapping.")
    return {key: _copy_runtime_value(item) for key, item in value.items()}


def _optional_bool_value(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean.")
    return value


class PydanticAITacticConfig(BaseModel):
    """Adapter options. Extra fields are left for application code."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    input_mode: InputMode = "auto"
    result_mode: ResultMode = "output"
    input_type: Any = None
    output_type: Any = None
    stream_mode: StreamMode = "output"
    include_context_metadata: StrictBool = True
    run_kwargs: dict[str, Any] = Field(default_factory=dict)
    input_mapper: Callable[[Any], Any] | None = None
    output_mapper: Callable[[Any], Any] | None = None
    description: StrictStr | None = None
    package_ref: StrictStr | None = None
    service_ref: StrictStr | None = None
    examples: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.run_kwargs = _copy_runtime_kwargs(self.run_kwargs)
        self.examples = copy_boundary_value(self.examples)
        self.metadata = copy_boundary_value(self.metadata)


class PydanticAITactic(Tactic[Any, Any]):
    """Expose a Pydantic AI agent as an LLLM tactic."""

    runtime_kind = "pydantic-ai"

    def __init__(
        self,
        agent: Any,
        config: Any = None,
        *,
        name: str | None = None,
        input_type: Any = None,
        output_type: Any = None,
        description: str | None = None,
        package_ref: str | None = None,
        service_ref: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        input_mode: InputMode | None = None,
        result_mode: ResultMode | None = None,
        stream_mode: StreamMode | None = None,
        include_context_metadata: bool | None = None,
        run_kwargs: dict[str, Any] | None = None,
        input_mapper: Callable[[Any], Any] | None = None,
        output_mapper: Callable[[Any], Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        cfg = (
            config
            if isinstance(config, PydanticAITacticConfig)
            else PydanticAITacticConfig.model_validate({} if config is None else config)
        )
        self.agent = agent
        self.input_mode = input_mode or cfg.input_mode
        self.result_mode = result_mode or cfg.result_mode
        self.stream_mode = stream_mode or cfg.stream_mode
        context_metadata = _optional_bool_value(
            include_context_metadata,
            "include_context_metadata",
        )
        self.include_context_metadata = (
            cfg.include_context_metadata
            if context_metadata is None
            else context_metadata
        )
        self.run_kwargs = _copy_runtime_kwargs(cfg.run_kwargs if run_kwargs is None else run_kwargs)
        self.input_mapper = input_mapper or cfg.input_mapper
        self.output_mapper = output_mapper or cfg.output_mapper
        self.input_type = input_type or cfg.input_type or self.input_type
        self.output_type = output_type or cfg.output_type or getattr(agent, "output_type", None)
        agent_description = description if description is not None else cfg.description
        if agent_description is None:
            agent_description = getattr(agent, "description", None) or inspect.getdoc(agent) or ""
        tactic_name = name if name is not None else getattr(agent, "name", None)
        super().__init__(
            name=tactic_name,
            description=agent_description,
            package_ref=package_ref if package_ref is not None else cfg.package_ref,
            service_ref=service_ref if service_ref is not None else cfg.service_ref,
            examples=examples if examples is not None else list(cfg.examples),
            metadata=metadata if metadata is not None else cfg.metadata,
        )

    @classmethod
    def from_agent(cls, agent: Any, **kwargs: Any) -> "PydanticAITactic":
        return cls(agent, **kwargs)

    def _run(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        method = getattr(self.agent, "run_sync", None)
        if method is None:
            raise TypeError("Pydantic AI agent must define run_sync() for sync calls.")
        task = self._map_input(input_value)
        result = method(task, **self._merged_kwargs(kwargs, context=context, method=method))
        return self._map_output(result)

    async def _arun(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        method = getattr(self.agent, "run", None)
        if method is None:
            return await super()._arun(input_value, context=context, **kwargs)
        task = self._map_input(input_value)
        result = method(task, **self._merged_kwargs(kwargs, context=context, method=method))
        if inspect.isawaitable(result):
            result = await result
        return self._map_output(result)

    def stream(self, input_value: Any, *, context: CallContext | None = None, **kwargs: Any):
        method = getattr(self.agent, "run_stream_sync", None)
        if method is None:
            return super().stream(input_value, context=context, **kwargs)
        task = self._map_input(input_value)
        result = method(task, **self._merged_kwargs(kwargs, context=context, method=method))
        return _iter_stream_result(result, mode=self.stream_mode)

    async def astream(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ):
        method = getattr(self.agent, "run_stream", None)
        if method is not None:
            task = self._map_input(input_value)
            result = method(task, **self._merged_kwargs(kwargs, context=context, method=method))
            if inspect.isawaitable(result):
                result = await result
            async for item in _aiter_stream_result(result, mode=self.stream_mode):
                yield item
            return
        sync_method = getattr(self.agent, "run_stream_sync", None)
        if sync_method is not None:
            task = self._map_input(input_value)
            result = sync_method(
                task,
                **self._merged_kwargs(kwargs, context=context, method=sync_method),
            )
            async for item in _aiter_stream_result(result, mode=self.stream_mode):
                yield item
            return
        async for item in super().astream(input_value, context=context, **kwargs):
            yield item

    async def aevents(self, input_value: Any, *, context: CallContext | None = None, **kwargs: Any):
        method = getattr(self.agent, "run_stream_events", None)
        if method is None:
            async for event in super().aevents(input_value, context=context, **kwargs):
                yield event
            return
        task = self._map_input(input_value)
        result = method(task, **self._merged_kwargs(kwargs, context=context, method=method))
        if inspect.isawaitable(result):
            result = await result
        async for item in _aiter_any(result):
            yield item

    def info(self) -> TacticInfo:
        info = super().info()
        output_schema = _agent_output_json_schema(self.agent)
        if output_schema is not None:
            info.output_schema = output_schema
        return info

    def capabilities(self) -> set[str]:
        supported = {"run", "arun"}
        if _has_static_callable(self.agent, "run_stream") or _has_static_callable(
            self.agent,
            "run_stream_sync",
        ):
            supported.add("stream")
        if _has_static_callable(self.agent, "run_stream_events"):
            supported.add("events")
        return supported

    def _merged_kwargs(
        self,
        kwargs: dict[str, Any],
        *,
        context: CallContext | None,
        method: Callable[..., Any],
    ) -> dict[str, Any]:
        merged = _copy_runtime_kwargs(self.run_kwargs)
        merged.update(_copy_runtime_kwargs(kwargs))
        if (
            self.include_context_metadata
            and context is not None
            and "metadata" not in merged
            and _accepts_keyword(method, "metadata")
        ):
            merged["metadata"] = _context_metadata(context)
        return merged

    def _map_input(self, value: Any) -> Any:
        if self.input_mapper is not None:
            return self.input_mapper(value)
        if isinstance(value, BaseModel):
            if self.input_mode in {"auto", "json"}:
                return value.model_dump_json()
            if self.input_mode == "dict":
                return value.model_dump(mode="json")
            if self.input_mode == "python":
                return value
            if self.input_mode == "text":
                return str(value)
        if self.input_mode == "text":
            return value if isinstance(value, str) else str(value)
        return value

    def _map_output(self, result: Any) -> Any:
        if self.output_mapper is not None:
            return self.output_mapper(result)
        if self.result_mode == "result":
            return result
        for attr in ("output", "data"):
            if hasattr(result, attr):
                return getattr(result, attr)
        return result


def tactic_as_tool(
    tactic: Tactic[Any, Any],
    *,
    name: str | None = None,
    description: str | None = None,
    parameter_mode: Literal["task", "kwargs"] = "task",
) -> Callable[..., Any]:
    """Expose any LLLM tactic as a plain callable for runtime-owned tool APIs."""

    if parameter_mode not in {"task", "kwargs"}:
        raise ValueError("parameter_mode must be 'task' or 'kwargs'.")
    if name is not None and not name.strip():
        raise ValueError("tool name must be non-empty when provided.")
    tool_name = _safe_name(name if name is not None else tactic.tactic_name)
    input_schema = getattr(tactic, "input_type", None)
    output_schema = getattr(tactic, "output_type", None)

    if parameter_mode == "kwargs":

        def tool(**kwargs: Any) -> Any:
            return tactic.run(_kwargs_to_task(input_schema, kwargs))

        if _is_basemodel_type(input_schema):
            tool.__signature__ = _signature_from_model(input_schema, output_schema)  # type: ignore[attr-defined]
    else:

        def tool(task: Any) -> Any:
            return tactic.run(task)

        tool.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters=[
                inspect.Parameter(
                    "task",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=input_schema or Any,
                )
            ],
            return_annotation=output_schema or Any,
        )

    tool.__name__ = tool_name
    tool.__qualname__ = tool_name
    tool.__doc__ = description or tactic.info().description or f"Run {tactic.tactic_name}."
    return tool


async def _aiter_any(value: Any):
    if hasattr(value, "__aenter__"):
        async with value as stream:
            async for item in _aiter_any(stream):
                yield item
        return
    if hasattr(value, "__aiter__"):
        async for item in value:
            yield item
        return
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray, dict)):
        for item in value:
            yield item
        return
    yield value


def _iter_stream_result(value: Any, *, mode: StreamMode):
    if mode == "raw":
        yield value
        return
    stream = _select_stream(value, mode)
    if stream is not None:
        yield from stream
        return
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray, dict)):
        yield from value
        return
    yield value


async def _aiter_stream_result(value: Any, *, mode: StreamMode):
    if mode == "raw":
        yield value
        return
    if hasattr(value, "__aenter__"):
        async with value as stream:
            if stream is value and hasattr(stream, "__aiter__"):
                async for item in stream:
                    yield item
            else:
                async for item in _aiter_stream_result(stream, mode=mode):
                    yield item
        return
    stream = _select_stream(value, mode)
    if stream is not None:
        async for item in _aiter_any(stream):
            yield item
        return
    async for item in _aiter_any(value):
        yield item


def _select_stream(value: Any, mode: StreamMode):
    if mode == "text" and hasattr(value, "stream_text"):
        return value.stream_text()
    if mode == "response" and hasattr(value, "stream_response"):
        return value.stream_response()
    if hasattr(value, "stream_output"):
        return value.stream_output()
    return None


def _agent_output_json_schema(agent: Any) -> dict[str, Any] | None:
    if not _has_static_callable(agent, "output_json_schema"):
        return None
    method = getattr(agent, "output_json_schema", None)
    if not callable(method):
        return None
    try:
        schema = method()
    except Exception:
        return None
    return schema if isinstance(schema, dict) else None


def _has_static_callable(value: Any, name: str) -> bool:
    try:
        attribute = inspect.getattr_static(value, name)
    except AttributeError:
        return False
    descriptor_value = getattr(attribute, "__func__", attribute)
    return callable(descriptor_value)


def _kwargs_to_task(input_schema: Any, kwargs: dict[str, Any]) -> Any:
    if _is_basemodel_type(input_schema):
        return input_schema.model_validate(kwargs)
    return kwargs


def _signature_from_model(model: type[BaseModel], output_schema: Any) -> inspect.Signature:
    parameters: list[inspect.Parameter] = []
    for field_name, field in model.model_fields.items():
        default = inspect.Parameter.empty if field.is_required() else field.default
        parameters.append(
            inspect.Parameter(
                field_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=field.annotation or Any,
            )
        )
    return inspect.Signature(
        parameters=parameters,
        return_annotation=output_schema or Any,
    )


def _is_basemodel_type(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)


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


def _accepts_keyword(method: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    if name in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _safe_name(value: str) -> str:
    candidate = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    candidate = candidate.strip("_")
    if not candidate:
        return "lllm_tool"
    if candidate[0].isdigit():
        return f"tool_{candidate}"
    return candidate
