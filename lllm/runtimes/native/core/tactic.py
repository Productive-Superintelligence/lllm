"""
Tactics — the callable protocol boundary in LLLM.

``Tactic`` is intentionally small: it wraps a unit of machine behavior behind
``call()/acall()`` with string or Pydantic-model input/output. The implementation
inside that boundary can be native LLLM, Pydantic AI, a graph runtime, a robot
monitor, or ordinary Python.

``NativeTactic`` lives in ``lllm.native`` and is exposed here lazily for
backward compatibility.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import copy
import datetime as dt
import hashlib
import inspect
import logging
import sys
import threading
import traceback as tb
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import aclosing, contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    TYPE_CHECKING,
    Tuple,
    Union,
    overload,
)

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ....protocol.errors import TacticUnsupportedError
from ....protocol.events import TacticEvent
from ....protocol.tactic import InputT, OutputT, _call_context, iterate_in_thread
from ....protocol.tactic import Tactic as _ProtocolTactic

if TYPE_CHECKING:
    from .agent import Agent
    from .dialog import Message
    from .prompt import Prompt

logger = logging.getLogger(__name__)

#: The call session of the tactic currently running in this context. A tactic
#: called while another is running records its session into the caller's, so
#: sub-tactic calls and their cost roll up into the parent session.
_ACTIVE_SESSION: contextvars.ContextVar[Optional["TacticCallSession"]] = (
    contextvars.ContextVar("lllm_active_tactic_session", default=None)
)


@dataclass(frozen=True, init=False)
class TacticContext:
    """Platform context supplied by package/service builders.

    This keeps the protocol-layer constructor small while preserving backward
    compatibility with the older ``runtime=`` and ``tactic_path=`` arguments.
    Production tracing and observability should be provided by Pydantic
    AI/Logfire/OpenTelemetry or another runtime-specific observer.
    """

    registry: Any
    tactic_path: Optional[str] = None
    observer: Any = None
    metadata: Mapping[str, Any]

    def __init__(
        self,
        registry: Any = None,
        tactic_path: Optional[str] = None,
        observer: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        runtime: Any = None,
    ):
        object.__setattr__(
            self, "registry", registry if registry is not None else runtime
        )
        object.__setattr__(self, "tactic_path", tactic_path)
        object.__setattr__(self, "observer", observer)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    @property
    def runtime(self) -> Any:
        """Compatibility alias for older code that named the registry runtime."""

        return self.registry

    @classmethod
    def from_compat(
        cls,
        context: Optional["TacticContext"] = None,
        *,
        registry: Any = None,
        runtime: Any = None,
        tactic_path: Optional[str] = None,
        observer: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "TacticContext":
        base = context or cls()
        selected_registry = (
            registry
            if registry is not None
            else runtime
            if runtime is not None
            else base.registry
        )
        return cls(
            registry=selected_registry,
            tactic_path=tactic_path if tactic_path is not None else base.tactic_path,
            observer=observer if observer is not None else base.observer,
            metadata=metadata if metadata is not None else base.metadata,
        )


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    if isinstance(config, BaseModel):
        return getattr(config, key, default)
    return default


def _config_as_mapping(config: Any, label: str) -> Dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    if isinstance(config, BaseModel):
        return config.model_dump()
    raise TypeError(f"{label} must be a mapping or Pydantic BaseModel.")


def _task_to_string(task: Any) -> str:
    if isinstance(task, str):
        return task
    if isinstance(task, BaseModel):
        try:
            return task.model_dump_json()
        except Exception:  # e.g. bytes that are not UTF-8
            return repr(task)
    return repr(task)


def _zero_invoke_cost() -> Any:
    from .const import InvokeCost

    return InvokeCost()


@contextmanager
def _observe_tactic_call(
    observer: Any,
    *,
    tactic: "Tactic",
    session: "TacticCallSession",
    task: Any,
    session_name: str,
    tags: Optional[Dict[str, str]],
    metadata: Optional[Dict[str, Any]],
):
    if observer is None or not hasattr(observer, "observe_tactic"):
        yield None
        return

    try:
        span_cm = observer.observe_tactic(
            tactic=tactic,
            session=session,
            task=task,
            session_name=session_name,
            tags=tags,
            metadata=metadata,
        )
    except Exception:
        logger.warning("Tactic observer failed to create span.", exc_info=True)
        yield None
        return

    if span_cm is None:
        yield None
        return

    try:
        span = span_cm.__enter__()
    except Exception:
        logger.warning("Tactic observer failed to enter span.", exc_info=True)
        yield None
        return

    try:
        yield span
    except BaseException:
        exc_info = sys.exc_info()
        try:
            span_cm.__exit__(*exc_info)
        except Exception:
            logger.warning("Tactic observer failed to close span.", exc_info=True)
        raise
    else:
        try:
            span_cm.__exit__(None, None, None)
        except Exception:
            logger.warning("Tactic observer failed to close span.", exc_info=True)


def _notify_tactic_observer(observer: Any, event: str, **kwargs: Any) -> None:
    if observer is None:
        return
    callback = getattr(observer, f"on_tactic_{event}", None)
    if callback is None:
        return
    try:
        callback(**kwargs)
    except Exception:
        logger.warning("Tactic observer failed during %s callback.", event, exc_info=True)


# ---------------------------------------------------------------------------
# TacticCallSession — per-call diagnostics
# ---------------------------------------------------------------------------


class TacticCallSession(BaseModel):
    """
    Tracks one invocation of a tactic — every agent call, every sub-tactic
    call, total cost, and the final result.

    The tactic is stateless; all per-call data lives here.
    """

    tactic_name: str
    tactic_path: Optional[str] = (
        None  # stable ID: "{package_name}::{tactic_name}", e.g. "my_pkg::researcher"
    )

    state: str = "initial"

    agent_sessions: Dict[str, List[Any]] = Field(default_factory=dict)
    sub_tactic_sessions: Dict[str, List["TacticCallSession"]] = Field(
        default_factory=dict
    )

    delivery: Optional[Any] = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None  # full traceback when state == "failure"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def record_agent_call(self, agent_name: str, session: Any) -> None:
        if agent_name not in self.agent_sessions:
            self.agent_sessions[agent_name] = []
        self.agent_sessions[agent_name].append(session)

    def record_sub_tactic_call(
        self, tactic_name: str, session: "TacticCallSession"
    ) -> None:
        if tactic_name not in self.sub_tactic_sessions:
            self.sub_tactic_sessions[tactic_name] = []
        self.sub_tactic_sessions[tactic_name].append(session)

    def success(self, result: Any) -> None:
        self.state = "success"
        self.delivery = result

    def failure(self, error: Optional[Exception] = None) -> None:
        self.state = "failure"
        if error is not None:
            self.error = f"{type(error).__name__}: {error}"
            self.error_traceback = tb.format_exc()

    @property
    def agent_cost(self) -> Any:
        total = _zero_invoke_cost()
        for sessions in self.agent_sessions.values():
            for s in sessions:
                cost = getattr(s, "cost", None)
                if cost is not None:
                    total = total + cost
        return total

    @property
    def sub_tactic_cost(self) -> Any:
        total = _zero_invoke_cost()
        for sessions in self.sub_tactic_sessions.values():
            for s in sessions:
                cost = getattr(s, "total_cost", None)
                if cost is not None:
                    total = total + cost
        return total

    @property
    def total_cost(self) -> Any:
        return self.agent_cost + self.sub_tactic_cost

    @property
    def agent_call_count(self) -> int:
        return sum(len(ss) for ss in self.agent_sessions.values())

    @property
    def sub_tactic_call_count(self) -> int:
        return sum(len(ss) for ss in self.sub_tactic_sessions.values())

    def summary(self) -> Dict[str, Any]:
        return {
            "tactic": self.tactic_name,
            "state": self.state,
            "agent_calls": self.agent_call_count,
            "sub_tactic_calls": self.sub_tactic_call_count,
            "total_cost": str(self.total_cost),
        }


# ---------------------------------------------------------------------------
# Tactic
# ---------------------------------------------------------------------------


class Tactic(_ProtocolTactic[InputT, OutputT]):
    """The program of an agentic system.

    A tactic owns a task interface and orchestrates the work behind it. Write
    the program in :meth:`call` (``call(self, task)``); set ``agent_group`` and
    pass agent configs to get fresh native :class:`Agent` objects in
    ``self.agents`` on every call. Each call runs on its own copy of the
    tactic inside a :class:`TacticCallSession` that records agent calls,
    sub-tactic calls, and cost.

    ``Tactic`` is also a protocol tactic: it validates typed input and output
    (``input_type``/``output_type``, or the ``input_model``/``output_model``
    aliases), and exposes ``run``/``arun``/``info`` so it can be served,
    streamed, packaged, and used as a tool by any runtime. Subclasses may
    implement the protocol hook ``_run`` (or ``_arun``) instead of ``call``.
    """

    name: Optional[str] = None
    input_model: Optional[type[BaseModel]] = None
    output_model: Optional[type[BaseModel]] = None
    input_type: Any = None
    output_type: Any = None
    runtime_kind = "native"
    #: Names of agents built from ``agent_configs`` for every call.
    agent_group: Optional[List[str]] = None
    #: The method the native entry points (``tactic(x)``, ``acall``,
    #: ``bcall``, ``ccall``) run: ``call`` when a subclass writes one, else
    #: its protocol hook (``_run``, then ``_arun``). Set per subclass.
    _native_program: ClassVar[str] = "call"
    _runtime_kind_declared: ClassVar[bool] = False

    # -- Optional registration --------------------------------------------

    def __init_subclass__(
        cls,
        register: bool = True,
        runtime: Any = None,
        registry: Any = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        writes_call = cls.call is not Tactic.call
        writes_run = cls._run is not Tactic._run
        writes_arun = cls._arun is not Tactic._arun
        if writes_call or not (writes_run or writes_arun):
            cls._native_program = "call"
        else:
            cls._native_program = "_run" if writes_run else "_arun"
        if cls._native_program != "call" and cls.__dict__.get("agent_group"):
            warnings.warn(
                f"{cls.__name__} declares agent_group but implements "
                f"{cls._native_program}() instead of call(): fresh agents are built "
                "only for call() programs and streams, so its hook would share the "
                "agents built at construction. Write call() instead.",
                stacklevel=2,
            )
        if cls.__dict__.get("agent_group") or (cls.agent_group and (
            "astream" in cls.__dict__ or "aevents" in cls.__dict__
        )):
            async_streams = [
                name
                for name in ("astream", "aevents")
                if name in cls.__dict__
                and inspect.isasyncgenfunction(inspect.unwrap(cls.__dict__[name]))
            ]
            if async_streams:
                warnings.warn(
                    f"{cls.__name__} declares agent_group and overrides "
                    f"{' and '.join(f'{name}()' for name in async_streams)}: async "
                    "streams run on the instance, so they would share the agents "
                    "built at construction. Override the sync events() or stream() "
                    "instead, which get fresh agents per stream.",
                    stacklevel=2,
                )
        if "runtime_kind" in cls.__dict__:
            cls._runtime_kind_declared = True
        elif not cls._runtime_kind_declared:
            # A tactic that only implements the protocol hooks (_run/_arun) is
            # plain Python; one that writes a call() program runs on the
            # native runtime. An explicit runtime_kind is inherited as is.
            protocol_only = cls._native_program != "call" and not cls.agent_group
            cls.runtime_kind = "python" if protocol_only else "native"
        # input_model/output_model alias input_type/output_type; a class's own
        # declaration wins over the types it inherits.
        for model_attr, type_attr in (
            ("input_model", "input_type"),
            ("output_model", "output_type"),
        ):
            model = getattr(cls, model_attr, None)
            if model is None:
                continue
            if model_attr in cls.__dict__ and type_attr not in cls.__dict__:
                setattr(cls, type_attr, model)
            elif getattr(cls, type_attr, None) is None:
                setattr(cls, type_attr, model)
        if (
            register
            and (runtime is not None or registry is not None)
            and getattr(cls, "name", None)
        ):
            from .tactic_registry import register_tactic_class

            register_tactic_class(cls, runtime=runtime, registry=registry)

    # -- Construction -----------------------------------------------------

    def __init__(
        self,
        config: Optional[Any] = None,
        runtime: Any = None,
        registry: Any = None,
        tactic_path: Optional[str] = None,
        context: Optional[TacticContext] = None,
        observer: Any = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        package_ref: Optional[str] = None,
        service_ref: Optional[str] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        self._sub_tactics: Dict[str, Tactic] = {}
        super().__init__(
            name=name,
            description=description,
            package_ref=package_ref,
            service_ref=service_ref,
            examples=examples,
            metadata=metadata,
        )
        self.context = TacticContext.from_compat(
            context,
            registry=registry,
            runtime=runtime,
            tactic_path=tactic_path,
            observer=observer,
        )
        self._registry = self.context.registry
        self.config = {} if config is None else config
        self._tactic_path: Optional[str] = self.context.tactic_path
        self._max_workers: int = _config_get(self.config, "max_workers", 4)
        self._session: Optional[TacticCallSession] = None
        self.call_context: Optional[Any] = None
        self._llm_invoker = None
        self._agent_specs: Dict[str, Any] = {}
        self.agents: Dict[str, Any] = {}
        if self.agent_group:
            self._init_agents()

    def capabilities(self) -> set[str]:
        """Protocol surfaces plus the native ``call``/``acall`` entry points."""

        cls = type(self)
        supported = super().capabilities() | {"call", "acall"}
        # This class overrides aevents()/astream() to serve events()/stream(),
        # so the protocol's checks always see an override; judge against it.
        supported -= {"events", "stream"}
        if cls.events is not Tactic.events or cls.aevents is not Tactic.aevents:
            supported.add("events")
        if cls.stream is not _ProtocolTactic.stream or cls.astream is not Tactic.astream:
            supported.add("stream")
        return supported

    def input_schema(self) -> Any:
        """Return the declared input schema/type, if any."""

        return self.input_type or self.input_model

    def output_schema(self) -> Any:
        """Return the declared output schema/type, if any."""

        return self.output_type or self.output_model

    # -- Sub-tactic composition -------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Tactic) and name not in ("_sub_tactics",):
            if hasattr(self, "_sub_tactics"):
                self._sub_tactics[name] = value
        super().__setattr__(name, value)

    @property
    def sub_tactics(self) -> Dict[str, "Tactic"]:
        return dict(self._sub_tactics)

    @property
    def tactic_name(self) -> str:
        return getattr(self, "_name", None) or self.name or type(self).__name__

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        parameter_mode: str = "task",
    ):
        """Expose this tactic as a typed Python callable.

        Pydantic AI and other agent runtimes can consume ordinary Python
        callables as tools. The implementation is imported lazily so the core
        tactic protocol stays small and adapter-owned.
        """

        from lllm.runtimes.pydantic_ai import tactic_as_tool

        return tactic_as_tool(
            self,
            name=name,
            description=description,
            parameter_mode=parameter_mode,
        )

    # -- Core execution ---------------------------------------------------

    def call(self, task: Any, **kwargs) -> Any:
        """The tactic's program. Override this, or the protocol hook ``_run``.

        ``tactic(task)``, ``acall``, ``bcall`` and ``ccall`` run ``call()``
        when a subclass writes it and its ``_run()`` (or ``_arun()``) hook
        otherwise, so both styles work through every entry point.
        """

        program = type(self)._native_program
        if program != "call":
            raise TacticUnsupportedError(
                f"{type(self).__name__} implements {program}() instead of call(), "
                "so call() is not available; run it with tactic(task), run() or arun()."
            )
        raise TacticUnsupportedError(
            f"{type(self).__name__} must implement call() or _run()."
        )

    def events(self, task: Any, **kwargs) -> Any:
        """Return a sync event stream for tactics that support it."""

        raise TacticUnsupportedError(f"{type(self).__name__} does not support events().")

    async def aevents(
        self, input_value: Any, *, context: Any = None, **kwargs: Any
    ) -> Any:
        """Yield the events of an overridden :meth:`events`, else of ``astream()``."""

        if type(self).events is Tactic.events:
            async with aclosing(
                super().aevents(input_value, context=context, **kwargs)
            ) as events:
                async for event in events:
                    yield event
            return
        context = _call_context(context)
        # Validate like run() does, so events() receives the typed input.
        validated = self.validate_input(input_value)
        if _accepts_context(type(self).events):
            kwargs.setdefault("context", context)
        async with aclosing(
            iterate_in_thread(
                lambda: self._scoped_stream("events", validated, context, kwargs)
            )
        ) as items:
            async for item in items:
                yield item if isinstance(item, TacticEvent) else TacticEvent(data=item)

    async def astream(
        self, input_value: Any, *, context: Any = None, **kwargs: Any
    ) -> Any:
        """Yield the items of an overridden ``stream()``, run like :meth:`call`."""

        if type(self).stream is _ProtocolTactic.stream:
            async with aclosing(
                super().astream(input_value, context=context, **kwargs)
            ) as items:
                async for item in items:
                    yield item
            return
        context = _call_context(context)
        validated = self.validate_input(input_value)
        kwargs["context"] = context
        async with aclosing(
            iterate_in_thread(
                lambda: self._scoped_stream("stream", validated, context, kwargs)
            )
        ) as items:
            async for item in items:
                yield item

    def _scoped_stream(
        self, method: str, task: Any, context: Any, kwargs: Dict[str, Any]
    ) -> Any:
        """Iterate ``events()``/``stream()`` on a per-call copy, like ``call()``.

        The copy gets fresh agents and its own session, so every stream (and
        every ``/stream`` request) starts with clean dialogs.
        """

        with self._call_scope(task, context) as (ctx, session, session_name):
            observer = ctx.context.observer
            with _observe_tactic_call(
                observer,
                tactic=ctx,
                session=session,
                task=task,
                session_name=session_name,
                tags=None,
                metadata=None,
            ) as span:
                try:
                    yield from getattr(ctx, method)(task, **kwargs)
                except GeneratorExit:
                    # Closed before the end (a consumer stopped, a client left).
                    session.state = "cancelled"
                    raise
                except Exception as exc:
                    ctx._record_failure(session, observer, span, task, exc, None, None)
                    raise
                ctx._record_success(session, observer, span, task, None, None, None)

    def run(
        self,
        input_value: Any,
        *,
        context: Any = None,
        return_trace: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run the tactic synchronously through the validated boundary.

        A tactic that only implements ``_arun()`` runs ``arun()`` to
        completion, so it has a sync entry point too.
        """

        if type(self)._native_program == "_arun":
            return _resolve_awaitable(
                self.arun(
                    input_value, context=context, return_trace=return_trace, **kwargs
                )
            )
        return super().run(
            input_value, context=context, return_trace=return_trace, **kwargs
        )

    def _run(self, input_value: Any, *, context: Any = None, **kwargs: Any) -> Any:
        """Protocol hook: run :meth:`call` inside a native call session.

        The native hooks never call each other's overrides: a subclass hook
        that hands its call back (``super()._run()``) reaches ``call()``,
        which raises, so any recursion goes through a public entry point.
        """

        kwargs.pop("return_session", None)
        return self._execute(
            input_value, context=context, validate=False, _program="call", **kwargs
        )

    async def _arun(
        self, input_value: Any, *, context: Any = None, **kwargs: Any
    ) -> Any:
        """Protocol hook: await an async ``call()`` here, run a sync one off the loop."""

        program = type(self)._native_program
        if program == "_arun":
            # Reached only through super()._arun() from an _arun() hook: there
            # is no sync program to fall back to.
            raise TacticUnsupportedError(
                f"{type(self).__name__} must implement call() or _run()."
            )
        kwargs.pop("return_session", None)
        if program == "call" and type(self)._program_is_async("call"):
            return await self._aexecute(
                input_value, context=context, validate=False, _program="call", **kwargs
            )
        return await asyncio.to_thread(self._run, input_value, context=context, **kwargs)

    @classmethod
    def _program_is_async(cls, program: str) -> bool:
        if program == "_arun":
            return True
        method = cls.call if program == "call" else cls._run
        return inspect.iscoroutinefunction(inspect.unwrap(method))

    def _copy_for_call(self) -> "Tactic":
        ctx = copy.copy(self)
        ctx._sub_tactics = dict(self._sub_tactics)
        return ctx

    def _prepare_session(self, session: TacticCallSession) -> None:
        """Give the per-call copy fresh agents that report into ``session``."""

        if not self.agent_group:
            return None
        from ..native.tactic import _TrackedAgent

        self.agents = {
            name: _TrackedAgent(agent, session, name)
            for name, agent in self._create_fresh_agents().items()
        }
        return None

    # -- Native agents ----------------------------------------------------

    @property
    def llm_invoker(self):
        if self._llm_invoker is None:
            from ..native.tactic import _build_native_invoker

            config = _config_as_mapping(self.config, "Native tactic config")
            self._llm_invoker = _build_native_invoker(config)
        return self._llm_invoker

    @llm_invoker.setter
    def llm_invoker(self, value):
        self._llm_invoker = value

    def _init_agents(self) -> None:
        from .native_config import parse_agent_configs

        config = _config_as_mapping(self.config, "Native tactic config")
        self._agent_specs = parse_agent_configs(config, self.agent_group, self.name)
        self.agents = self._create_fresh_agents()

    def _create_fresh_agents(self) -> Dict[str, Any]:
        if not self._agent_specs:
            return {}
        from ..native.tactic import _get_default_registry

        runtime = self._registry or _get_default_registry()
        return {
            agent_name: spec.build(runtime, self.llm_invoker)
            for agent_name, spec in self._agent_specs.items()
        }

    def _resolve_tactic_path(self, tactic_name: str) -> str:
        if self._tactic_path is not None:
            return self._tactic_path

        if self.name is not None and self._registry is not None:
            try:
                node = self._registry.get_node(self.name, resource_type="tactic")
                from .tactic_registry import _stable_tactic_id

                self._tactic_path = _stable_tactic_id(node.namespace, self.name)
                return self._tactic_path
            except (KeyError, AttributeError):
                pass

        self._tactic_path = self.name or tactic_name
        return self._tactic_path

    def _execute(
        self,
        task: Any,
        session_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        return_session: bool = False,
        context: Any = None,
        validate: bool = True,
        *,
        _program: Optional[str] = None,
        **kwargs,
    ) -> Any:
        program = _program or type(self)._native_program
        # call() programs run on a per-call copy with fresh agents; protocol
        # hooks run on the instance, as they do under run() and arun().
        with self._call_scope(
            task, context, session_name=session_name, per_call_copy=program == "call"
        ) as (ctx, session, session_name):
            result = ctx._run_observed(
                task,
                kwargs,
                session=session,
                session_name=session_name,
                tags=tags,
                metadata=metadata,
                context=context,
                validate=validate,
                program=program,
            )
        return session if return_session else result

    async def _aexecute(
        self,
        task: Any,
        session_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        return_session: bool = False,
        context: Any = None,
        validate: bool = True,
        *,
        _program: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """:meth:`_execute` for async programs, awaited on the caller's loop."""

        program = _program or type(self)._native_program
        with self._call_scope(
            task, context, session_name=session_name, per_call_copy=program == "call"
        ) as (ctx, session, session_name):
            result = await ctx._arun_observed(
                task,
                kwargs,
                session=session,
                session_name=session_name,
                tags=tags,
                metadata=metadata,
                context=context,
                validate=validate,
                program=program,
            )
        return session if return_session else result

    @contextmanager
    def _call_scope(
        self,
        task: Any,
        context: Any,
        *,
        session_name: Optional[str] = None,
        per_call_copy: bool = True,
    ):
        """Open a native call: a session that sub-tactic calls roll up into.

        With ``per_call_copy`` the program runs on a copy of the tactic that
        has fresh agents reporting into the session (the v1 design); without
        it, on the instance itself. Yields ``(tactic, session, session_name)``.
        """

        tactic_name = self.tactic_name
        if session_name is None:
            task_hash = hashlib.md5(_task_to_string(task).encode()).hexdigest()[:8]
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"{tactic_name}_{task_hash}_{timestamp}"
        session = TacticCallSession(
            tactic_name=tactic_name,
            tactic_path=self._resolve_tactic_path(tactic_name),
        )
        session.state = "running"
        ctx = self
        if per_call_copy:
            ctx = self._copy_for_call()
            ctx.call_context = context
            ctx._session = session
            ctx._prepare_session(session)
        parent_session = _ACTIVE_SESSION.get()
        session_token = _ACTIVE_SESSION.set(session)
        logger.info("Tactic '%s' started — session_name=%s", tactic_name, session_name)
        try:
            yield ctx, session, session_name
        finally:
            _ACTIVE_SESSION.reset(session_token)
            if parent_session is not None and parent_session is not session:
                parent_session.record_sub_tactic_call(tactic_name, session)

    def _run_observed(
        self,
        task: Any,
        kwargs: Dict[str, Any],
        *,
        session: "TacticCallSession",
        session_name: str,
        tags: Optional[Dict[str, str]],
        metadata: Optional[Dict[str, Any]],
        context: Any,
        validate: bool,
        program: str,
    ) -> Any:
        observer = self.context.observer
        with _observe_tactic_call(
            observer,
            tactic=self,
            session=session,
            task=task,
            session_name=session_name,
            tags=tags,
            metadata=metadata,
        ) as span:
            try:
                validated = self.validate_input(task) if validate else task
                # An async program called from a sync entry point runs on the
                # shared background loop.
                result = _resolve_awaitable(
                    self._invoke_program(program, validated, context, kwargs)
                )
                result = self.validate_output(result) if validate else result
            except Exception as exc:
                self._record_failure(session, observer, span, task, exc, tags, metadata)
                raise
            self._record_success(session, observer, span, validated, result, tags, metadata)
        return result

    async def _arun_observed(
        self,
        task: Any,
        kwargs: Dict[str, Any],
        *,
        session: "TacticCallSession",
        session_name: str,
        tags: Optional[Dict[str, str]],
        metadata: Optional[Dict[str, Any]],
        context: Any,
        validate: bool,
        program: str,
    ) -> Any:
        observer = self.context.observer
        with _observe_tactic_call(
            observer,
            tactic=self,
            session=session,
            task=task,
            session_name=session_name,
            tags=tags,
            metadata=metadata,
        ) as span:
            try:
                validated = self.validate_input(task) if validate else task
                result = self._invoke_program(program, validated, context, kwargs)
                if inspect.isawaitable(result):
                    result = await result
                result = self.validate_output(result) if validate else result
            except Exception as exc:
                self._record_failure(session, observer, span, task, exc, tags, metadata)
                raise
            self._record_success(session, observer, span, validated, result, tags, metadata)
        return result

    def _invoke_program(
        self, program: str, task: Any, context: Any, kwargs: Dict[str, Any]
    ) -> Any:
        """Call the program; an async one returns an awaitable."""

        kwargs = dict(kwargs)
        if program == "call":
            if context is not None and _accepts_context(type(self).call):
                kwargs.setdefault("context", context)
            return self.call(task, **kwargs)
        if program == "_run":
            return self._run(task, context=_call_context(context), **kwargs)
        if program == "_arun":
            return self._arun(task, context=_call_context(context), **kwargs)
        raise ValueError(f"Unknown tactic program: {program!r}")

    def _record_success(
        self, session, observer, span, task, result, tags, metadata
    ) -> None:
        session.success(result)
        _notify_tactic_observer(
            observer,
            "success",
            tactic=self,
            session=session,
            task=task,
            result=result,
            span=span,
            tags=tags,
            metadata=metadata,
        )
        logger.info(
            "Tactic '%s' completed — cost=%s agent_calls=%d",
            session.tactic_name,
            session.total_cost.cost,
            session.agent_call_count,
        )

    def _record_failure(self, session, observer, span, task, exc, tags, metadata) -> None:
        session.failure(exc)
        _notify_tactic_observer(
            observer,
            "failure",
            tactic=self,
            session=session,
            task=task,
            exception=exc,
            span=span,
            tags=tags,
            metadata=metadata,
        )
        # The exception reaches the caller and the session and observer record
        # it, so keep the traceback out of the default logs.
        logger.info(
            "Tactic '%s' failed: %s: %s", session.tactic_name, type(exc).__name__, exc
        )
        logger.debug("Tactic '%s' failure traceback:", session.tactic_name, exc_info=True)

    def __call__(
        self,
        task,
        session_name=None,
        tags=None,
        metadata=None,
        return_session=False,
        **kwargs,
    ):
        return self._execute(
            task,
            session_name,
            tags=tags,
            metadata=metadata,
            return_session=return_session,
            **kwargs,
        )

    async def acall(
        self, task, tags=None, metadata=None, return_session=False, **kwargs
    ):
        if type(self)._program_is_async(type(self)._native_program):
            return await self._aexecute(
                task,
                tags=tags,
                metadata=metadata,
                return_session=return_session,
                **kwargs,
            )
        loop = asyncio.get_running_loop()
        run_in_context = contextvars.copy_context().run
        return await loop.run_in_executor(
            None,
            run_in_context,
            lambda: self._execute(
                task,
                tags=tags,
                metadata=metadata,
                return_session=return_session,
                **kwargs,
            ),
        )

    def bcall(
        self,
        tasks,
        max_workers=None,
        fail_fast=True,
        tags=None,
        metadata=None,
        return_sessions=False,
        **kwargs,
    ):
        workers = max_workers or self._max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    contextvars.copy_context().run,
                    self._execute,
                    t,
                    None,
                    tags,
                    metadata,
                    return_session=return_sessions,
                    **kwargs,
                )
                for t in tasks
            ]
            if fail_fast:
                return [f.result() for f in futures]
            results = []
            for f in futures:
                try:
                    results.append(f.result())
                except Exception as e:
                    results.append(e)
            return results

    async def ccall(
        self,
        tasks,
        max_workers=None,
        tags=None,
        metadata=None,
        return_sessions=False,
        **kwargs,
    ):
        workers = max_workers or self._max_workers
        if type(self)._program_is_async(type(self)._native_program):
            limit = asyncio.Semaphore(workers)

            async def run_one(idx, t):
                async with limit:
                    return idx, await self._aexecute(
                        t,
                        tags=tags,
                        metadata=metadata,
                        return_session=return_sessions,
                        **kwargs,
                    )

            pending = [asyncio.ensure_future(run_one(i, t)) for i, t in enumerate(tasks)]
            try:
                for next_done in asyncio.as_completed(pending):
                    yield await next_done
            finally:
                for task_future in pending:
                    task_future.cancel()
            return
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:

            def _run(idx, t):
                return idx, self._execute(
                    t,
                    tags=tags,
                    metadata=metadata,
                    return_session=return_sessions,
                    **kwargs,
                )

            futures = [
                loop.run_in_executor(pool, contextvars.copy_context().run, _run, i, t)
                for i, t in enumerate(tasks)
            ]
            for coro in asyncio.as_completed(futures):
                idx, result = await coro
                yield idx, result

    # -- Quick constructor ------------------------------------------------

    @overload
    @classmethod
    def quick(
        cls,
        query: None = None,
        system_prompt: Optional[Union[str, Prompt]] = "You are a helpful assistant.",
        model: str = "gpt-4o",
        return_agent: bool = False,
        **model_args: Any,
    ) -> Agent: ...

    @overload
    @classmethod
    def quick(
        cls,
        query: str,
        system_prompt: Optional[Union[str, Prompt]] = "You are a helpful assistant.",
        model: str = "gpt-4o",
        return_agent: Literal[False] = False,
        **model_args: Any,
    ) -> Message: ...

    @overload
    @classmethod
    def quick(
        cls,
        query: str,
        system_prompt: Optional[Union[str, Prompt]] = "You are a helpful assistant.",
        model: str = "gpt-4o",
        return_agent: Literal[True] = True,
        **model_args: Any,
    ) -> Tuple[Message, Agent]: ...

    @classmethod
    def quick(
        cls,
        query: Optional[str] = None,
        system_prompt: Optional[Union[str, Prompt]] = "You are a helpful assistant.",
        model: str = "gpt-4o",
        return_agent: bool = False,
        **model_args: Any,
    ) -> Union[Message, Agent, Tuple[Message, Agent]]:
        """One agent, no configuration: ``Tactic.quick("What is LLLM?")``.

        With ``query`` it returns the reply (and the agent when
        ``return_agent=True``); without ``query`` it returns the agent so you
        can ``open``/``receive``/``respond`` yourself.
        """

        from ..native.tactic import _build_native_invoker, _get_default_registry
        from .agent import Agent
        from .prompt import Prompt

        if isinstance(system_prompt, str):
            prompt = Prompt(path="_quick/system", prompt=system_prompt)
        else:
            prompt = system_prompt
        agent = Agent(
            name="assistant",
            system_prompt=prompt,
            model=model,
            llm_invoker=_build_native_invoker({"invoker": "litellm"}),
            runtime=_get_default_registry(),
            model_args=model_args,
        )
        if query is None:
            return agent
        agent.open("chat")
        agent.receive(query)
        response = agent.respond()
        return (response, agent) if return_agent else response

    def __repr__(self) -> str:
        parts = [f"{type(self).__name__}(name={self.name!r}"]
        if self._agent_specs:
            parts.append(f"agents={list(self._agent_specs.keys())}")
        if self._sub_tactics:
            parts.append(f"sub_tactics={list(self._sub_tactics.keys())}")
        return ", ".join(parts) + ")"


def __getattr__(name: str) -> Any:
    if name == "tactictool":
        from .tactic_tool import tactictool

        return tactictool
    if name in {"NativeTactic", "_TrackedAgent", "_build_native_invoker"}:
        from ..native import tactic as native_tactic

        return getattr(native_tactic, name)
    if name in {
        "_normalize_name",
        "_stable_tactic_id",
        "build_tactic",
        "get_tactic_class",
        "register_tactic_class",
    }:
        from . import tactic_registry

        return getattr(tactic_registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _accepts_context(fn: Any) -> bool:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "context" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


_BACKGROUND_LOOP: Optional[asyncio.AbstractEventLoop] = None
_BACKGROUND_THREAD: Optional[threading.Thread] = None
_BACKGROUND_LOCK = threading.Lock()


def _background_loop() -> Tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """The event loop that runs async programs for sync callers.

    One loop for the process, so clients and pools an async program keeps
    between calls (an ``httpx.AsyncClient``, say) stay bound to a live loop.
    """

    global _BACKGROUND_LOOP, _BACKGROUND_THREAD
    with _BACKGROUND_LOCK:
        if _BACKGROUND_LOOP is None or _BACKGROUND_LOOP.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="lllm-async-programs", daemon=True
            )
            thread.start()
            _BACKGROUND_LOOP, _BACKGROUND_THREAD = loop, thread
        return _BACKGROUND_LOOP, _BACKGROUND_THREAD


def _resolve_awaitable(value: Any) -> Any:
    """Return ``value``, first running it to completion if it is awaitable.

    Awaitables run on the shared background loop (see :func:`_background_loop`)
    with the caller's context variables, so sub-tactic calls still roll up
    into the active session. Async code should await ``acall()``/``arun()``
    instead, which run async programs on its own loop.
    """

    if not inspect.isawaitable(value):
        return value
    loop, thread = _background_loop()
    if threading.current_thread() is thread:
        if inspect.iscoroutine(value):
            value.close()
        raise TacticUnsupportedError(
            "An async tactic program called a sync entry point of another async "
            "program; use `await tactic.acall(...)` (or `await tactic.arun(...)`) "
            "from async code."
        )

    async def wait() -> Any:
        return await value

    result: concurrent.futures.Future = concurrent.futures.Future()

    def start() -> None:
        # Created inside the caller's context, so the task copies it.
        task = loop.create_task(wait())

        def done(finished: "asyncio.Task[Any]") -> None:
            if finished.cancelled():
                result.cancel()
            elif finished.exception() is not None:
                result.set_exception(finished.exception())
            else:
                result.set_result(finished.result())

        task.add_done_callback(done)

    loop.call_soon_threadsafe(start, context=contextvars.copy_context())
    return result.result()
