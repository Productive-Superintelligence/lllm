"""Strategy abstraction for application packages."""

from __future__ import annotations

import functools
import inspect
import threading
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from ..protocol import CallContext, Tactic
from .models import ResourceKind, StrategyInfo, StrategyResource, _segment

Handler = Callable[..., Any | Awaitable[Any]]


class ResourceNotFoundError(KeyError):
    """Raised when a strategy has no resource with the requested name."""

    def __str__(self) -> str:
        # KeyError renders its argument with repr(); keep the message plain.
        return str(self.args[0]) if self.args else ""


class StrategyStore:
    """A strategy's channel store, opened on first use.

    It wraps a store you attach, or names a local store folder that is created
    only when the store is first used, so building or inspecting a strategy
    never touches disk. The strategy declares its channels here; they are
    created in the store (existing channels are left as they are) the first
    time any store method is called. Call store methods on this object
    directly; :meth:`open` returns the underlying store. Threads may share
    it: the store is opened, and each channel created, once.
    """

    def __init__(self, store: Any = None, *, root: str | Path | None = None) -> None:
        if (store is None) == (root is None):
            raise ValueError("StrategyStore takes exactly one of store or root.")
        self._store = store
        if root is None:
            root = getattr(store, "root", None)
        #: The store folder, or None for a store without one (a remote store).
        self.root = None if root is None else Path(root).expanduser().resolve()
        self._pending: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def declare(self, channel: Mapping[str, Any]) -> None:
        """Create ``channel``, a ``create_channel`` payload, on the next use."""

        with self._lock:
            self._pending[channel["name"]] = dict(channel)

    def open(self) -> Any:
        """Open the store, create the declared channels, and return the store."""

        store = self._store
        if store is not None and not self._pending:
            return store
        # One thread opens the store and creates the channels while the others
        # wait, so no caller gets the store before its channels exist.
        with self._lock:
            if self._store is None:
                from ..network.stores import LocalStore

                self._store = LocalStore(self.root)
            for name, channel in list(self._pending.items()):
                _create_channel(self._store, channel)
                del self._pending[name]
            return self._store

    def same_as(self, store: Any) -> bool:
        """Whether ``store`` is this store: the same object or the same folder."""

        if store is self:
            return True
        if isinstance(store, StrategyStore):
            other, root = store._store, store.root
        else:
            other, root = store, getattr(store, "root", None)
        if other is not None and other is self._store:
            return True
        return root is not None and self.root is not None and Path(root).resolve() == self.root

    def adopt(self, store: Any) -> bool:
        """Use ``store``, a store object for this folder, if none is open yet.

        If this store has not opened its folder yet, it uses ``store`` (or the
        store it wraps) instead of opening a new local store. Returns whether
        ``store`` is the store in use: ``False`` if another store object is.
        """

        other = store._store if isinstance(store, StrategyStore) else store
        with self._lock:
            if self._store is None:
                self._store = other
            return other is None or other is self._store

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.open(), name)

    def __repr__(self) -> str:
        return f"StrategyStore({self._store if self._store is not None else self.root!r})"


class Strategy:
    """Encapsulate package resources and expose one application-level boundary."""

    def __init__(
        self,
        name: str,
        *,
        description: str = "",
        resources: list[StrategyResource] | tuple[StrategyResource, ...] = (),
        runner: Handler | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        # The rule StrategyInfo checks, so info() and GET /strategy cannot fail on it.
        self.name = _segment(name, "strategy.name")
        self.description = description
        self._resources: dict[str, StrategyResource] = {}
        self._handlers: dict[str, Handler] = {}
        self._tactics: dict[str, Any] = {}
        # Store channels behind the channel resources that have no handler.
        self._channels: dict[str, dict[str, Any]] = {}
        self._store: StrategyStore | None = None
        self._runner = runner
        self.metadata = dict(metadata or {})
        for resource in resources:
            self.add_resource(resource)

    @property
    def store(self) -> StrategyStore | None:
        """The channel store behind this strategy's channel resources, if any.

        A :class:`StrategyStore`: it opens the underlying store on first use.
        Assigning to it is the same as calling :meth:`attach_store`.
        """

        return self._store

    @store.setter
    def store(self, store: Any) -> None:
        self.attach_store(store)

    @property
    def resources(self) -> tuple[StrategyResource, ...]:
        return tuple(self._resources.values())

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name=self.name,
            description=self.description,
            resources=self.resources,
            metadata=dict(self.metadata),
        )

    def runner(self, fn: Handler) -> Handler:
        """Decorator or setter for the strategy's main run handler."""

        self._runner = fn
        return fn

    def add_resource(
        self,
        resource: StrategyResource,
        *,
        handler: Handler | None = None,
    ) -> "Strategy":
        """Add a resource; ``handler`` makes it callable.

        A channel resource without a handler is backed by :attr:`store`, see
        :meth:`attach_store`.
        """

        channel = None
        if handler is None and resource.kind == "channel":
            channel = {"description": resource.description}
        return self._add_resource(resource, handler=handler, channel=channel)

    def _add_resource(
        self,
        resource: StrategyResource,
        *,
        handler: Handler | None = None,
        channel: Mapping[str, Any] | None = None,
    ) -> "Strategy":
        """Add ``resource``; ``channel`` declares the store channel behind it."""

        if resource.name in self._resources:
            raise ValueError(f"Strategy resource already exists: {resource.name}")
        self._resources[resource.name] = resource
        if channel is not None:
            declared = {**channel, "name": resource.name}
            self._channels[resource.name] = declared
            if self._store is not None:
                self._store.declare(declared)
        if handler is None:
            return self
        if resource.kind == "tactic":
            self._tactics[resource.name] = handler
        elif isinstance(handler, Tactic):
            self._handlers[resource.name] = _tactic_handler(handler)
        else:
            self._handlers[resource.name] = handler
        return self

    def add_tactic(
        self,
        tactic: Tactic[Any, Any],
        *,
        name: str | None = None,
        ref: str | None = None,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Strategy":
        """Mount an lllm tactic; it is served at ``/tactics/<name>/``."""

        info = tactic.info()
        return self.add_resource(
            StrategyResource(
                kind="tactic",
                name=name or info.name,
                ref=ref or info.package_ref,
                description=info.description if description is None else description,
                metadata={"runtime": info.runtime_kind, **dict(metadata or {})},
            ),
            handler=tactic,
        )

    def attach_store(self, store: Any) -> "Strategy":
        """Back channel resources with a store; it is served at ``/network/``.

        A strategy has one store. Every channel resource without a handler is
        backed by the store channel of the same name, created in the store the
        first time the store is used. Packages that declare channels share the
        strategy's store (see :func:`lllm.shell.add_package`), so attach your
        store before mounting them. A store object for the folder of a package
        store is used as the strategy's store if the strategy has not opened
        that folder yet (it does when it is served or a channel is used).
        Attaching the store the strategy already has, or ``None`` to a strategy
        without a store, does nothing. Attaching another store, another store
        object for a folder the strategy has opened, or ``None`` to a strategy
        that has a store raises ``ValueError``: a store is never replaced or
        dropped silently.
        """

        if self._store is not None:
            if not self._store.same_as(store):
                change = "drop it" if store is None else f"also use {_store_label(store)}"
                raise ValueError(
                    f"Strategy {self.name!r} already has a channel store "
                    f"({_store_label(self._store)}) and cannot {change}: a strategy "
                    "has one store, served at /network/. Attach it before mounting "
                    "packages, and mount packages without store= to share it."
                )
            if not self._store.adopt(store):
                raise ValueError(
                    f"Strategy {self.name!r} already uses another store object for "
                    f"its channel store ({_store_label(self._store)}) and cannot "
                    f"switch to this {type(store).__name__}: a strategy has one "
                    "store, served at /network/. Attach your store before the "
                    "strategy is served or its channels are used."
                )
            return self
        if store is None:
            return self
        self._store = store if isinstance(store, StrategyStore) else StrategyStore(store)
        for channel in self._channels.values():
            self._store.declare(channel)
        return self

    def tactics(self) -> dict[str, Tactic[Any, Any]]:
        """Tactic resources as lllm tactics.

        Tactic objects are returned as-is, lazy package tactics are loaded, and
        plain handler functions are wrapped with :func:`lllm.as_tactic`. A
        handler that takes no input is called without it, as the strategy
        calls its other handlers.
        """

        return {name: self._tactic(name) for name in self._tactics}

    def _tactic(self, name: str) -> Tactic[Any, Any]:
        value = self._tactics[name]
        if isinstance(value, Tactic):
            return value
        if callable(getattr(value, "load", None)):
            return value.load()
        from ..runtimes import as_tactic

        return as_tactic(_tactic_callable(value), name=name)

    def _lookup(self, name: str) -> StrategyResource:
        resource = self._resources.get(name)
        if resource is None:
            raise ResourceNotFoundError(f"Strategy resource not found: {name}")
        return resource

    def resource(
        self,
        kind: ResourceKind,
        name: str,
        *,
        ref: str | None = None,
        entrypoint: str | None = None,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
        handler: Handler | None = None,
    ) -> "Strategy":
        return self.add_resource(
            StrategyResource(
                kind=kind,
                name=name,
                ref=ref,
                entrypoint=entrypoint,
                description=description,
                metadata=dict(metadata or {}),
            ),
            handler=handler,
        )

    def tactic(self, name: str, **kwargs: Any) -> "Strategy":
        return self.resource("tactic", name, **kwargs)

    def channel(self, name: str, **kwargs: Any) -> "Strategy":
        return self.resource("channel", name, **kwargs)

    def service(self, name: str, **kwargs: Any) -> "Strategy":
        return self.resource("service", name, **kwargs)

    def package(self, name: str, **kwargs: Any) -> "Strategy":
        return self.resource("package", name, **kwargs)

    def snapshot(self, name: str, **kwargs: Any) -> "Strategy":
        return self.resource("snapshot", name, **kwargs)

    def use_package(
        self,
        path: str | Path,
        *,
        prefix: str | None = None,
        store: str | Path | None = None,
        bind: bool = True,
    ) -> "Strategy":
        """Import resources from a LLLM hub package manifest."""

        from .packages import add_package

        return add_package(self, path, prefix=prefix, store=store, bind=bind)

    async def arun(
        self,
        input_value: Any = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        if self._runner is None:
            return {
                "strategy": self.name,
                "input": input_value,
                "resources": [resource.model_dump() for resource in self.resources],
            }
        return await _call(self._runner, input_value, context=context)

    async def invoke_resource(
        self,
        name: str,
        input_value: Any = None,
        *,
        context: Mapping[str, Any] | CallContext | None = None,
    ) -> Any:
        """Call the resource ``name``; what ``context`` means depends on it.

        - A tactic resource with a handler runs through its tactic exactly as
          ``POST /tactics/<name>/run`` runs it: ``context`` has the shape of a
          :class:`~lllm.protocol.CallContext` (``{"metadata": {...}, "tags":
          {...}, "request_id": ...}``), or is one, and the tactic receives a
          ``CallContext``. A plain-function handler is wrapped with
          :func:`lllm.as_tactic`, so it gets a ``CallContext`` on both routes
          if it takes ``context``; one that takes no input is called without.
        - Any other resource with a handler receives ``context`` as a plain
          dict (a ``Tactic`` handler gets that dict as ``CallContext.metadata``).
        - A channel resource without a handler is served by :attr:`store`:
          ``input_value`` holds an ``op``, one of ``append``, ``query`` (the
          default), or ``info``, and that op's fields.
        - Any other resource without a handler echoes the resource record, the
          input, and the context.

        Raises :class:`ResourceNotFoundError`, a ``KeyError``, for unknown names.
        """

        resource = self._lookup(name)
        if name in self._tactics:
            tactic = self._tactic(name)
            return await tactic.arun(input_value, context=_tactic_context(tactic, context))
        handler = self._handlers.get(name)
        if handler is None and name in self._channels and self._store is not None:
            handler = _channel_handler(self._store, name)
        if handler is None:
            return {
                "resource": resource.model_dump(),
                "input": input_value,
                "context": dict(context or {}),
            }
        return await _call(handler, input_value, context=context)


async def _call(
    handler: Handler,
    input_value: Any,
    *,
    context: Mapping[str, Any] | None,
) -> Any:
    kwargs: dict[str, Any] = {}
    signature = _signature(handler)
    if signature is not None and _accepts_context(signature):
        kwargs["context"] = dict(context or {})
    if signature is not None and _can_accept_input(signature):
        result = handler(input_value, **kwargs)
    else:
        result = handler(**kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _tactic_handler(tactic: Tactic[Any, Any]) -> Handler:
    async def handler(
        input_value: Any = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        call_context = None if context is None else CallContext(metadata=dict(context))
        return await tactic.arun(input_value, context=call_context)

    return handler


def _tactic_callable(handler: Handler) -> Handler:
    """``handler`` in the shape :func:`lllm.as_tactic` calls: ``handler(input, context=...)``.

    ``as_tactic`` always passes the input and awaits only coroutine functions,
    while the strategy calls its other handlers without the input if they take
    none and awaits what they return. So a tactic handler that takes no input
    (``lambda: ...``, ``lambda *, context=None: ...``) is called without it, and
    an object with an ``async def __call__`` is awaited. Other handlers are
    returned as they are.
    """

    signature = _signature(handler)
    takes_input = signature is None or _can_accept_input(signature)
    async_object = not inspect.iscoroutinefunction(handler) and inspect.iscoroutinefunction(
        getattr(type(handler), "__call__", None)
    )
    if takes_input and not async_object:
        return handler

    def call(input_value: Any, kwargs: dict[str, Any]) -> Any:
        return handler(input_value, **kwargs) if takes_input else handler(**kwargs)

    # wraps() keeps the handler's name, docs, and signature, which as_tactic
    # reads to describe the tactic and to decide whether to pass ``context``.
    if async_object or inspect.iscoroutinefunction(handler):

        @functools.wraps(handler)
        async def adapted(input_value: Any = None, **kwargs: Any) -> Any:
            return await call(input_value, kwargs)

    else:

        @functools.wraps(handler)
        def adapted(input_value: Any = None, **kwargs: Any) -> Any:
            return call(input_value, kwargs)

    return adapted


def _tactic_context(tactic: Tactic[Any, Any], context: Any) -> CallContext:
    """The ``CallContext`` that ``POST /tactics/<name>/run`` builds from ``context``."""

    if isinstance(context, CallContext):
        return context
    from ..services.fastapi import _context_from_data

    return _context_from_data(context, tactic=tactic, endpoint="run")


def _channel_handler(store: Any, name: str) -> Handler:
    async def handler(
        input_value: Any = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        request = input_value if isinstance(input_value, Mapping) else {}
        op = str(request.get("op") or "query")
        if op == "info":
            return store.get_channel(name)
        if op == "append":
            event = {
                "channel": name,
                "payload": request.get("payload"),
                "kind": request.get("kind") or "event",
                "source": request.get("source") or "lllm.shell",
                "schema": request.get("schema"),
                "metadata": request.get("metadata") or {},
                "correlation_id": request.get("correlation_id"),
                "parent_ids": tuple(request.get("parent_ids") or ()),
            }
            return store.append_event(event)
        if op == "query":
            return list(
                store.query_events(
                    name,
                    after_cursor=_non_negative_int(request.get("after_cursor", 0)),
                    limit=_positive_int(request.get("limit", 100)),
                    kind=request.get("kind"),
                )
            )
        raise ValueError("channel op must be one of: append, info, query")

    return handler


def _create_channel(store: Any, channel: dict[str, Any]) -> None:
    create = getattr(store, "create_channel", None)
    if create is None or inspect.iscoroutinefunction(create):
        return  # async stores cannot be prepared here; create their channels yourself
    from ..network.core import ChannelExistsError

    try:
        create(channel)
    except ChannelExistsError:
        pass
    except Exception as exc:
        # A RemoteStore reports an existing channel as an HTTP 409 error.
        if getattr(exc, "error_type", None) != "ChannelExistsError":
            raise


def _store_label(store: Any) -> str:
    root = getattr(store, "root", None)
    if root is not None:
        return str(root)
    if isinstance(store, StrategyStore):
        store = store._store
    return type(store).__name__


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("after_cursor must be a non-negative integer")
    return value


def _positive_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("limit must be a positive integer")
    return value


def _signature(handler: Handler) -> inspect.Signature | None:
    try:
        return inspect.signature(handler)
    except (TypeError, ValueError):
        return None


def _accepts_context(signature: inspect.Signature) -> bool:
    if "context" in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _can_accept_input(signature: inspect.Signature) -> bool:
    # The input is passed positionally, so keyword-only parameters cannot take it.
    for parameter in signature.parameters.values():
        if parameter.name == "context":
            continue
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            return True
    return any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
