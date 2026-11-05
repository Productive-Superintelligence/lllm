"""Run tactics over channel events.

A :class:`TacticProcessor` subscribes to a source channel, feeds each event
payload to a tactic, and appends the tactic's output to a sink channel. Output
events point back at their source event through ``parent_ids`` and keep its
``correlation_id``, so a chain of processors stays traceable.

It works with any store that speaks the channel API: a
:class:`~lllm.network.LocalStore`, a :class:`~lllm.network.RemoteStore`, or
(with :meth:`TacticProcessor.aprocess`) an :class:`~lllm.network.AsyncRemoteStore`.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal

from pydantic import BaseModel, TypeAdapter

from ..protocol import CallContext, Tactic
from .core import Event, SubscriptionNotFoundError


class TacticProcessor:
    """Feed a channel's events through a tactic.

    ``source`` events are read through a named subscription, so a processor
    resumes where it stopped across restarts. Each event is pulled and
    acknowledged on its own; with ``errors="raise"`` a failing event stops the
    run without skipping the events behind it, and with ``errors="record"`` the
    failure is written to ``sink`` as a ``kind="error"`` event and processing
    continues.
    """

    def __init__(
        self,
        tactic: Tactic[Any, Any],
        store: Any,
        *,
        source: str,
        sink: str | None = None,
        consumer: str | None = None,
        kind: str | None = None,
        output_kind: str = "result",
        errors: Literal["raise", "record"] = "raise",
    ) -> None:
        if errors not in {"raise", "record"}:
            raise ValueError("errors must be 'raise' or 'record'.")
        self.tactic = tactic
        self.store = store
        self.source = source
        self.sink = sink
        self.consumer = consumer or f"{tactic.name}-processor"
        self.output_kind = output_kind
        self.errors = errors
        self.kind = kind
        self.subscription_id = self.consumer
        self._subscribed = False

    def process(self, limit: int | None = None) -> list[Event | Any]:
        """Process pending events; return sink events (or raw outputs without a sink)."""

        self._subscribe()
        results: list[Event | Any] = []
        while limit is None or len(results) < limit:
            events = self.store.pull_subscription(self.subscription_id, limit=1)
            if not events:
                break
            event = events[0]
            try:
                output = self.tactic.run(event.payload, context=self._context(event))
            except Exception as exc:
                results.append(self.store.append_event(self._error_event(event, exc)))
                continue
            results.append(self._emit(event, output))
        return results

    async def aprocess(self, limit: int | None = None) -> list[Event | Any]:
        """Async variant of :meth:`process` using ``tactic.arun``."""

        await self._asubscribe()
        results: list[Event | Any] = []
        while limit is None or len(results) < limit:
            events = await _maybe_await(
                self.store.pull_subscription(self.subscription_id, limit=1)
            )
            if not events:
                break
            event = events[0]
            try:
                output = await self.tactic.arun(
                    event.payload, context=self._context(event)
                )
            except Exception as exc:
                error_event = self._error_event(event, exc)
                results.append(await _maybe_await(self.store.append_event(error_event)))
                continue
            output_event = self._output_event(event, output)
            if output_event is None:
                results.append(output)
            else:
                results.append(await _maybe_await(self.store.append_event(output_event)))
        return results

    def _subscribe(self) -> None:
        if self._subscribed:
            return
        try:
            self.store.get_subscription(self.subscription_id)
        except Exception as exc:
            if not _is_missing(exc):
                raise
            self.store.create_subscription(self.source, **self._subscription_args())
        self._subscribed = True

    async def _asubscribe(self) -> None:
        if self._subscribed:
            return
        try:
            await _maybe_await(self.store.get_subscription(self.subscription_id))
        except Exception as exc:
            if not _is_missing(exc):
                raise
            await _maybe_await(
                self.store.create_subscription(self.source, **self._subscription_args())
            )
        self._subscribed = True

    def _subscription_args(self) -> dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "consumer": self.consumer,
            "filters": {"kind": self.kind} if self.kind else None,
        }

    def _context(self, event: Event) -> CallContext:
        return CallContext(
            caller=self.consumer,
            tags={"channel": self.source},
            metadata={
                "event_id": event.id,
                "event_kind": event.kind,
                "correlation_id": event.correlation_id or event.id,
            },
        )

    def _emit(self, event: Event, output: Any) -> Event | Any:
        output_event = self._output_event(event, output)
        if output_event is None:
            return output
        return self.store.append_event(output_event)

    def _output_event(self, event: Event, output: Any) -> Event | None:
        if self.sink is None:
            return None
        return Event(
            channel=self.sink,
            source=self.consumer,
            kind=self.output_kind,
            payload=_payload(output),
            correlation_id=event.correlation_id or event.id,
            parent_ids=(event.id,),
        )

    def _error_event(self, event: Event, exc: Exception) -> Event:
        if self.errors == "raise" or self.sink is None:
            raise exc
        return Event(
            channel=self.sink,
            source=self.consumer,
            kind="error",
            payload={"error": str(exc), "type": type(exc).__name__},
            correlation_id=event.correlation_id or event.id,
            parent_ids=(event.id,),
        )


def _payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    try:
        return TypeAdapter(type(value)).dump_python(value, mode="json")
    except Exception:
        return value


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _is_missing(exc: Exception) -> bool:
    if isinstance(exc, SubscriptionNotFoundError):
        return True
    return getattr(exc, "status_code", None) == 404
