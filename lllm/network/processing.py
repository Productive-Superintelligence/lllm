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
from .core import Event, Subscription, SubscriptionNotFoundError


class TacticProcessor:
    """Feed a channel's events through a tactic.

    ``source`` events are read through a named subscription, so a processor
    resumes where it stopped across restarts. ``consumer`` names that
    subscription and defaults to ``"<tactic name>-processor"``; a tactic whose
    name contains whitespace, ``?``, or ``#`` needs an explicit ``consumer``.

    Events are processed one at a time and at least once: the tactic runs on
    an event, and its output is turned into a sink event, before the event is
    acknowledged (the subscription cursor moves past it). The sink event is
    appended only after that, so a failed append raises and is not retried.
    ``errors`` decides what a failure in the tactic or in building the sink
    event does. With ``errors="raise"`` the failing event stays unacknowledged
    and the exception propagates, so the next call runs that same event again
    before any event behind it. With ``errors="record"`` the event is
    acknowledged, the failure is written to ``sink`` as a ``kind="error"``
    event, and processing continues; without a ``sink`` it acts like
    ``"raise"``. A tactic can see an event more than once (after a failure or
    an interrupted run), so it should tolerate repeats.

    Run one worker per ``consumer``; the guarantees above assume it. Workers
    sharing a consumer can process the same event twice, and can lose one:
    when another worker acknowledges the event this processor ran, the
    processor drops its output for it, and its own acknowledgement covers the
    next pending event instead, which it runs right away. That event is
    already acknowledged, so with ``errors="raise"`` a failure loses it.
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
        self.consumer = consumer or _default_consumer(tactic)
        self.output_kind = output_kind
        self.errors = errors
        self.kind = kind
        self.subscription_id = self.consumer
        self._subscribed = False

    def process(self, limit: int | None = None) -> list[Event | Any]:
        """Process pending events; return sink events (or raw outputs without a sink).

        Each event is peeked at the subscription cursor, run through the
        tactic, acknowledged, and only then written to ``sink``. ``limit`` caps
        the number of results, outputs and recorded errors alike. This method
        needs a synchronous store; use :meth:`aprocess` with async stores such
        as :class:`~lllm.network.AsyncRemoteStore`.
        """

        if _is_async_store(self.store):
            raise TypeError(
                f"process() cannot use the async store {type(self.store).__name__}; "
                "use `await processor.aprocess()` with async stores."
            )
        self._subscribe()
        results: list[Event | Any] = []
        while limit is None or len(results) < limit:
            event = self._peek()
            if event is None:
                break
            output, sink_event = self._call(event)
            acked = self.store.pull_subscription(self.subscription_id, limit=1)
            if not acked:
                break
            if acked[0].id != event.id:
                # Another worker acknowledged ``event`` first, so its output is
                # theirs to write. The event this pull acknowledged is ours now;
                # it is already acknowledged, so a raised failure is not retried.
                event = acked[0]
                output, sink_event = self._call(event)
            results.append(
                output if sink_event is None else self.store.append_event(sink_event)
            )
        return results

    async def aprocess(self, limit: int | None = None) -> list[Event | Any]:
        """Async variant of :meth:`process` using ``tactic.arun``.

        It works with synchronous and asynchronous stores, with the same
        at-least-once semantics.
        """

        await self._asubscribe()
        results: list[Event | Any] = []
        while limit is None or len(results) < limit:
            event = await self._apeek()
            if event is None:
                break
            output, sink_event = await self._acall(event)
            acked = await _maybe_await(
                self.store.pull_subscription(self.subscription_id, limit=1)
            )
            if not acked:
                break
            if acked[0].id != event.id:
                # Another worker acknowledged ``event`` first, so its output is
                # theirs to write. The event this pull acknowledged is ours now;
                # it is already acknowledged, so a raised failure is not retried.
                event = acked[0]
                output, sink_event = await self._acall(event)
            results.append(
                output
                if sink_event is None
                else await _maybe_await(self.store.append_event(sink_event))
            )
        return results

    def _peek(self) -> Event | None:
        subscription = self.store.get_subscription(self.subscription_id)
        events = self.store.query_events(
            subscription.channel,
            after_cursor=subscription.cursor,
            limit=1,
            kind=subscription.filters.get("kind"),
        )
        return events[0] if events else None

    async def _apeek(self) -> Event | None:
        subscription = await _maybe_await(
            self.store.get_subscription(self.subscription_id)
        )
        events = await _maybe_await(
            self.store.query_events(
                subscription.channel,
                after_cursor=subscription.cursor,
                limit=1,
                kind=subscription.filters.get("kind"),
            )
        )
        return events[0] if events else None

    def _call(self, event: Event) -> tuple[Any, Event | None]:
        try:
            output = self.tactic.run(event.payload, context=self._context(event))
            return output, self._output_event(event, output)
        except Exception as exc:
            return None, self._error_event(event, exc)

    async def _acall(self, event: Event) -> tuple[Any, Event | None]:
        try:
            output = await self.tactic.arun(event.payload, context=self._context(event))
            return output, self._output_event(event, output)
        except Exception as exc:
            return None, self._error_event(event, exc)

    def _subscribe(self) -> None:
        if self._subscribed:
            return
        try:
            subscription = self.store.get_subscription(self.subscription_id)
        except Exception as exc:
            if not _is_missing(exc):
                raise
            subscription = self.store.create_subscription(
                self.source, **self._subscription_args()
            )
        self._check_subscription(subscription)
        self._subscribed = True

    async def _asubscribe(self) -> None:
        if self._subscribed:
            return
        try:
            subscription = await _maybe_await(
                self.store.get_subscription(self.subscription_id)
            )
        except Exception as exc:
            if not _is_missing(exc):
                raise
            subscription = await _maybe_await(
                self.store.create_subscription(self.source, **self._subscription_args())
            )
        self._check_subscription(subscription)
        self._subscribed = True

    def _check_subscription(self, subscription: Subscription) -> None:
        kind = subscription.filters.get("kind")
        if subscription.channel == self.source and kind == (self.kind or None):
            return
        raise ValueError(
            f"Subscription {self.subscription_id!r} already reads channel "
            f"{subscription.channel!r} with kind filter {kind!r}, but this "
            f"processor reads channel {self.source!r} with kind filter "
            f"{self.kind or None!r}. Pass a distinct consumer= to give this "
            "processor its own subscription."
        )

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


def _default_consumer(tactic: Tactic[Any, Any]) -> str:
    # Tactic names may hold characters a subscription id cannot: whitespace is
    # not a valid id, and "?" or "#" would cut a RemoteStore URL short.
    name = tactic.tactic_name
    if any(ch.isspace() or ch in "?#" for ch in name):
        raise ValueError(
            f"Cannot derive a consumer from tactic name {name!r}: a consumer "
            "name cannot contain whitespace, and '?' or '#' would break "
            "RemoteStore URLs. Pass consumer= to name this processor's "
            "subscription."
        )
    return f"{name}-processor"


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


def _is_async_store(store: Any) -> bool:
    return inspect.iscoroutinefunction(getattr(store, "pull_subscription", None))


def _is_missing(exc: Exception) -> bool:
    if isinstance(exc, SubscriptionNotFoundError):
        return True
    return getattr(exc, "status_code", None) == 404
