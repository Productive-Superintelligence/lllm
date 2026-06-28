"""Runtime-agnostic proxy utilities for tactics."""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field

from .protocol import CallContext, Tactic, TacticEvent, TacticUnsupportedError

ValueHook = Callable[[Any, CallContext], Any]
ErrorHook = Callable[[BaseException, CallContext], Any]
RecordSink = Callable[["ProxyRecord"], None]


class ProxyRecord(BaseModel):
    """One observed proxy call."""

    request_id: str
    proxy: str
    tactic: str
    state: Literal["success", "failure"]
    started_at: float
    ended_at: float
    latency_ms: float
    input_value: Any = None
    output_value: Any = None
    error_type: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InMemoryProxyLog:
    """Simple proxy record sink for tests, examples, and local tools."""

    def __init__(self) -> None:
        self.records: list[ProxyRecord] = []

    def append(self, record: ProxyRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()


class ProxyTactic(Tactic[Any, Any]):
    """Wrap another tactic with small before/after/error hooks.

    The proxy sits at the public ``Tactic`` boundary, so it can wrap plain
    Python, Pydantic AI, native, or remote tactics without knowing their
    runtime internals.
    """

    runtime_kind = "proxy"

    def __init__(
        self,
        tactic: Tactic[Any, Any],
        *,
        name: str | None = None,
        description: str | None = None,
        before: ValueHook | None = None,
        after: ValueHook | None = None,
        on_error: ErrorHook | None = None,
        sink: RecordSink | None = None,
        capture_inputs: bool = False,
        capture_outputs: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.tactic = tactic
        self.before = before
        self.after = after
        self.on_error = on_error
        self.sink = sink
        self.capture_inputs = capture_inputs
        self.capture_outputs = capture_outputs
        self.proxy_metadata = dict(metadata or {})
        info = tactic.info()
        self.input_type = tactic.input_type
        self.output_type = tactic.output_type
        super().__init__(
            name=name or f"{tactic.tactic_name}_proxy",
            description=description or f"Proxy for {tactic.tactic_name}.",
            package_ref=tactic.package_ref,
            service_ref=tactic.service_ref,
            examples=info.examples,
            metadata={
                "proxied_tactic": tactic.tactic_name,
                "proxied_runtime_kind": info.runtime_kind,
                **self.proxy_metadata,
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
        started_at = time.time()
        proxied_input = input_value
        try:
            proxied_input = _call_value_hook(self.before, proxied_input, context)
            output = self.tactic.run(proxied_input, context=context, **kwargs)
            output = _call_value_hook(self.after, output, context)
        except Exception as exc:
            _call_error_hook(self.on_error, exc, context)
            self._record_failure(started_at, context, proxied_input, exc)
            raise
        self._record_success(started_at, context, proxied_input, output)
        return output

    async def _arun(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        context = context or CallContext()
        started_at = time.time()
        proxied_input = input_value
        try:
            proxied_input = await _acall_value_hook(self.before, proxied_input, context)
            output = await self.tactic.arun(proxied_input, context=context, **kwargs)
            output = await _acall_value_hook(self.after, output, context)
        except Exception as exc:
            await _acall_error_hook(self.on_error, exc, context)
            self._record_failure(started_at, context, proxied_input, exc)
            raise
        self._record_success(started_at, context, proxied_input, output)
        return output

    def stream(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        context = context or CallContext()
        started_at = time.time()
        proxied_input = input_value
        chunks: list[Any] = []
        try:
            proxied_input = _call_value_hook(self.before, proxied_input, context)
            for item in self.tactic.stream(proxied_input, context=context, **kwargs):
                item = _call_value_hook(self.after, item, context)
                if self.capture_outputs:
                    chunks.append(item)
                yield item
        except Exception as exc:
            _call_error_hook(self.on_error, exc, context)
            self._record_failure(started_at, context, proxied_input, exc)
            raise
        self._record_success(started_at, context, proxied_input, chunks)

    async def astream(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        context = context or CallContext()
        started_at = time.time()
        proxied_input = input_value
        chunks: list[Any] = []
        try:
            proxied_input = await _acall_value_hook(self.before, proxied_input, context)
            async for item in self.tactic.astream(proxied_input, context=context, **kwargs):
                item = await _acall_value_hook(self.after, item, context)
                if self.capture_outputs:
                    chunks.append(item)
                yield item
        except Exception as exc:
            await _acall_error_hook(self.on_error, exc, context)
            self._record_failure(started_at, context, proxied_input, exc)
            raise
        self._record_success(started_at, context, proxied_input, chunks)

    async def aevents(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[TacticEvent]:
        if not self.tactic.supports("events"):
            async for event in super().aevents(input_value, context=context, **kwargs):
                yield event
            return

        context = context or CallContext()
        started_at = time.time()
        proxied_input = input_value
        events: list[Any] = []
        try:
            proxied_input = await _acall_value_hook(self.before, proxied_input, context)
            async for event in self.tactic.aevents(proxied_input, context=context, **kwargs):
                event = await _acall_value_hook(self.after, event, context)
                if not isinstance(event, TacticEvent):
                    event = TacticEvent(data=event)
                if self.capture_outputs:
                    events.append(event)
                yield event
        except Exception as exc:
            await _acall_error_hook(self.on_error, exc, context)
            self._record_failure(started_at, context, proxied_input, exc)
            raise
        self._record_success(started_at, context, proxied_input, events)

    def capabilities(self) -> set[str]:
        supported = {"run", "arun"}
        if self.tactic.supports("stream"):
            supported.add("stream")
        if self.tactic.supports("events"):
            supported.add("events")
        return supported

    def _record_success(
        self,
        started_at: float,
        context: CallContext,
        input_value: Any,
        output_value: Any,
    ) -> None:
        self._emit(
            started_at,
            context,
            state="success",
            input_value=input_value,
            output_value=output_value,
        )

    def _record_failure(
        self,
        started_at: float,
        context: CallContext,
        input_value: Any,
        exc: BaseException,
    ) -> None:
        self._emit(
            started_at,
            context,
            state="failure",
            input_value=input_value,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    def _emit(
        self,
        started_at: float,
        context: CallContext,
        *,
        state: Literal["success", "failure"],
        input_value: Any,
        output_value: Any = None,
        error_type: str | None = None,
        error: str | None = None,
    ) -> None:
        if self.sink is None:
            return
        ended_at = time.time()
        self.sink(
            ProxyRecord(
                request_id=context.request_id,
                proxy=self.tactic_name,
                tactic=self.tactic.tactic_name,
                state=state,
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=(ended_at - started_at) * 1000,
                input_value=_portable(input_value) if self.capture_inputs else None,
                output_value=_portable(output_value) if self.capture_outputs else None,
                error_type=error_type,
                error=error,
                metadata={
                    "context": dict(context.metadata),
                    "proxy": dict(self.proxy_metadata),
                },
            )
        )


def proxy_tactic(tactic: Tactic[Any, Any], **kwargs: Any) -> ProxyTactic:
    """Return a proxy wrapper for *tactic*."""

    return ProxyTactic(tactic, **kwargs)


def _call_value_hook(
    hook: ValueHook | None,
    value: Any,
    context: CallContext,
) -> Any:
    if hook is None:
        return value
    result = hook(value, context)
    if inspect.isawaitable(result):
        raise TacticUnsupportedError("Async proxy hooks require arun().")
    return value if result is None else result


async def _acall_value_hook(
    hook: ValueHook | None,
    value: Any,
    context: CallContext,
) -> Any:
    if hook is None:
        return value
    result = hook(value, context)
    if inspect.isawaitable(result):
        result = await result
    return value if result is None else result


def _call_error_hook(
    hook: ErrorHook | None,
    exc: BaseException,
    context: CallContext,
) -> None:
    if hook is None:
        return
    result = hook(exc, context)
    if inspect.isawaitable(result):
        raise TacticUnsupportedError("Async proxy error hooks require arun().")


async def _acall_error_hook(
    hook: ErrorHook | None,
    exc: BaseException,
    context: CallContext,
) -> None:
    if hook is None:
        return
    result = hook(exc, context)
    if inspect.isawaitable(result):
        await result


def _portable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    return value


__all__ = [
    "ErrorHook",
    "InMemoryProxyLog",
    "ProxyRecord",
    "ProxyTactic",
    "RecordSink",
    "ValueHook",
    "proxy_tactic",
]
