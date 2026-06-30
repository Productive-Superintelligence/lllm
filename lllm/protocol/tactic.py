"""Runtime-agnostic tactic protocol."""

from __future__ import annotations

import asyncio
import inspect
import time
import traceback
from copy import deepcopy
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, Field, StrictStr, model_validator

from ._validation import optional_text_value, path_segment_value, token_value
from .context import CallContext
from .errors import TacticUnsupportedError
from .events import TacticEvent
from .schema import SchemaRef, export_json_schema, type_name, validate_with_schema

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class CallTrace(BaseModel):
    """Boundary-level trace for one tactic call."""

    request_id: StrictStr
    tactic: StrictStr
    state: StrictStr = "running"
    started_at: float = Field(default_factory=time.time)
    ended_at: float | None = None
    latency_ms: float | None = None
    input_type: StrictStr | None = None
    output_type: StrictStr | None = None
    error_type: StrictStr | None = None
    error: StrictStr | None = None
    traceback: StrictStr | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.metadata = deepcopy(self.metadata)

    @model_validator(mode="after")
    def _validate_identity(self) -> "CallTrace":
        token_value(self.request_id, "request_id")
        path_segment_value(self.tactic, "tactic")
        token_value(self.state, "state")
        if self.error_type is not None:
            token_value(self.error_type, "error_type")
        return self

    def success(self, output: Any) -> None:
        self.state = "success"
        self.ended_at = time.time()
        self.latency_ms = (self.ended_at - self.started_at) * 1000
        self.output_type = type(output).__name__

    def failure(self, exc: BaseException) -> None:
        self.state = "failure"
        self.ended_at = time.time()
        self.latency_ms = (self.ended_at - self.started_at) * 1000
        self.error_type = type(exc).__name__
        self.error = str(exc)
        self.traceback = traceback.format_exc()


class CallResult(BaseModel):
    """Optional local result envelope with trace data."""

    output: Any = None
    trace: CallTrace

    def model_post_init(self, __context: Any) -> None:
        self.output = deepcopy(self.output)
        self.trace = self.trace.model_copy(deep=True)


class TacticInfo(BaseModel):
    """Static description of a tactic boundary."""

    name: StrictStr
    description: StrictStr = ""
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    capabilities: tuple[StrictStr, ...] = ("run", "arun")
    runtime_kind: StrictStr = "python"
    package_ref: StrictStr | None = None
    service_ref: StrictStr | None = None
    examples: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.input_schema = deepcopy(self.input_schema)
        self.output_schema = deepcopy(self.output_schema)
        self.examples = deepcopy(self.examples)
        self.metadata = deepcopy(self.metadata)

    @model_validator(mode="after")
    def _validate_identity(self) -> "TacticInfo":
        path_segment_value(self.name, "name")
        token_value(self.runtime_kind, "runtime_kind")
        for capability in self.capabilities:
            token_value(capability, "capabilities")
        return self


class Tactic(Generic[InputT, OutputT]):
    """Typed, service-ready unit that does one thing well.

    Subclasses normally implement ``_run`` for synchronous tactics, ``_arun``
    for async-only tactics, or both for runtimes with specialized behavior.
    ``run`` and ``arun`` are the public validated boundary methods.
    """

    name: ClassVar[str | None] = None
    description: ClassVar[str] = ""
    input_type: ClassVar[SchemaRef | None] = None
    output_type: ClassVar[SchemaRef | None] = None
    runtime_kind: ClassVar[str] = "python"

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        package_ref: str | None = None,
        service_ref: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        name_value = name if name is not None else self.name or type(self).__name__
        self._name = path_segment_value(name_value, "name")
        self._description = optional_text_value(description, "description")
        self.package_ref = optional_text_value(package_ref, "package_ref")
        self.service_ref = optional_text_value(service_ref, "service_ref")
        self.examples = deepcopy(examples or [])
        self.metadata = _metadata_mapping(metadata)

    @property
    def tactic_name(self) -> str:
        return self._name

    def info(self) -> TacticInfo:
        """Return the tactic's stable public contract."""

        return TacticInfo(
            name=self.tactic_name,
            description=self._description
            if self._description is not None
            else self.description
            or inspect.getdoc(type(self))
            or "",
            input_schema=export_json_schema(self.input_type),
            output_schema=export_json_schema(self.output_type),
            capabilities=tuple(sorted(self.capabilities())),
            runtime_kind=self.runtime_kind,
            package_ref=self.package_ref,
            service_ref=self.service_ref,
            examples=deepcopy(self.examples),
            metadata=deepcopy(self.metadata),
        )

    def validate_input(self, value: Any) -> Any:
        return validate_with_schema(value, self.input_type)

    def validate_output(self, value: Any) -> Any:
        return validate_with_schema(value, self.output_type)

    def run(
        self,
        input_value: InputT,
        *,
        context: CallContext | None = None,
        return_trace: bool = False,
        **kwargs: Any,
    ) -> OutputT | CallResult:
        """Run the tactic synchronously through the validated boundary."""

        context = _call_context(context)
        trace = self._new_trace(context)
        try:
            validated = self.validate_input(input_value)
            output = self._run(validated, context=context, **kwargs)
            output = self.validate_output(output)
            trace.success(output)
        except Exception as exc:
            trace.failure(exc)
            if return_trace:
                return CallResult(output=None, trace=trace)
            raise
        return CallResult(output=output, trace=trace) if return_trace else output

    async def arun(
        self,
        input_value: InputT,
        *,
        context: CallContext | None = None,
        return_trace: bool = False,
        **kwargs: Any,
    ) -> OutputT | CallResult:
        """Run the tactic asynchronously through the validated boundary."""

        context = _call_context(context)
        trace = self._new_trace(context)
        try:
            validated = self.validate_input(input_value)
            output = self._arun(validated, context=context, **kwargs)
            if inspect.isawaitable(output):
                output = await output
            output = self.validate_output(output)
            trace.success(output)
        except Exception as exc:
            trace.failure(exc)
            if return_trace:
                return CallResult(output=None, trace=trace)
            raise
        return CallResult(output=output, trace=trace) if return_trace else output

    def __call__(self, input_value: InputT, **kwargs: Any) -> OutputT | CallResult:
        return self.run(input_value, **kwargs)

    def _run(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        raise TacticUnsupportedError(f"{type(self).__name__} does not implement _run().")

    async def _arun(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        return await asyncio.to_thread(self._run, input_value, context=context, **kwargs)

    def stream(
        self,
        input_value: InputT,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        raise TacticUnsupportedError(f"{type(self).__name__} does not support stream().")

    async def astream(
        self,
        input_value: InputT,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        context = _call_context(context)
        for item in self.stream(input_value, context=context, **kwargs):
            yield item

    async def aevents(
        self,
        input_value: InputT,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[TacticEvent]:
        context = _call_context(context)
        async for item in self.astream(input_value, context=context, **kwargs):
            if isinstance(item, TacticEvent):
                yield item
            else:
                yield TacticEvent(data=item)

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities()

    def capabilities(self) -> set[str]:
        supported = {"run", "arun"}
        if type(self).stream is not Tactic.stream or type(self).astream is not Tactic.astream:
            supported.add("stream")
        if type(self).aevents is not Tactic.aevents:
            supported.add("events")
        return supported

    def _new_trace(self, context: CallContext) -> CallTrace:
        return CallTrace(
            request_id=context.request_id,
            tactic=self.tactic_name,
            input_type=type_name(self.input_type),
            metadata=deepcopy(context.metadata),
        )


def _call_context(value: Any) -> CallContext:
    if value is None:
        return CallContext()
    if isinstance(value, CallContext):
        return value
    raise TypeError("context must be a CallContext.")


def _metadata_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping.")
    return deepcopy(dict(value))
