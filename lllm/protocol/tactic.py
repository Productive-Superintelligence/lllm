"""Runtime-agnostic tactic protocol."""

from __future__ import annotations

import asyncio
import inspect
import time
import traceback
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, Field

from .context import CallContext
from .errors import TacticUnsupportedError
from .events import TacticEvent
from .schema import SchemaRef, export_json_schema, type_name, validate_with_schema

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class CallTrace(BaseModel):
    """Boundary-level trace for one tactic call."""

    request_id: str
    tactic: str
    state: str = "running"
    started_at: float = Field(default_factory=time.time)
    ended_at: float | None = None
    latency_ms: float | None = None
    input_type: str | None = None
    output_type: str | None = None
    error_type: str | None = None
    error: str | None = None
    traceback: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

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


class TacticInfo(BaseModel):
    """Static description of a tactic boundary."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    capabilities: tuple[str, ...] = ("run", "arun")
    runtime_kind: str = "python"
    package_ref: str | None = None
    service_ref: str | None = None
    examples: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
        self._name = name or self.name or type(self).__name__
        self._description = description
        self.package_ref = package_ref
        self.service_ref = service_ref
        self.examples = list(examples or [])
        self.metadata = dict(metadata or {})

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
            examples=list(self.examples),
            metadata=dict(self.metadata),
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

        context = context or CallContext()
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

        context = context or CallContext()
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
        for item in self.stream(input_value, context=context, **kwargs):
            yield item

    async def aevents(
        self,
        input_value: InputT,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[TacticEvent]:
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
            metadata=dict(context.metadata),
        )
