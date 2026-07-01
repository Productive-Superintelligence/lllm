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
import copy
import datetime as dt
import hashlib
import inspect
import logging
import sys
import traceback as tb
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
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

if TYPE_CHECKING:
    from .agent import Agent
    from .dialog import Message
    from .prompt import Prompt

logger = logging.getLogger(__name__)


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
        return task.model_dump_json()
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


class Tactic(ABC):
    """
    Runtime-neutral protocol boundary for a callable agentic machine.

    A tactic accepts a task and returns a result. Inputs and outputs may be
    strings or Pydantic models, which makes the same boundary work naturally
    for Python packages, FastAPI services, Pydantic AI agents, native LLLM
    agents, graph runtimes, and plain Python systems.

    Native LLLM Agent/Prompt/Dialog orchestration is available through
    :class:`NativeTactic`.
    """

    name: Optional[str] = None
    input_model: Optional[type[BaseModel]] = None
    output_model: Optional[type[BaseModel]] = None
    input_type: Any = None
    output_type: Any = None

    # -- Optional registration --------------------------------------------

    def __init_subclass__(
        cls,
        register: bool = True,
        runtime: Any = None,
        registry: Any = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
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
    ):
        self.context = TacticContext.from_compat(
            context,
            registry=registry,
            runtime=runtime,
            tactic_path=tactic_path,
            observer=observer,
        )
        self._registry = self.context.registry
        self._sub_tactics: Dict[str, Tactic] = {}

        self.config = {} if config is None else config
        self._tactic_path: Optional[str] = self.context.tactic_path
        self._max_workers: int = _config_get(self.config, "max_workers", 4)
        self._session: Optional[TacticCallSession] = None

    def supports(self, capability: str) -> bool:
        """Return whether this tactic supports an optional protocol surface."""

        return capability in self.capabilities()

    def capabilities(self) -> set[str]:
        """Return optional protocol surfaces implemented by this tactic."""

        supported = {"call", "acall"}
        if (
            type(self).stream is not Tactic.stream
            or type(self).astream is not Tactic.astream
        ):
            supported.add("stream")
        if (
            type(self).events is not Tactic.events
            or type(self).aevents is not Tactic.aevents
        ):
            supported.add("events")
        return supported

    def input_schema(self) -> Any:
        """Return the declared input schema/type, if any."""

        return self.input_type or self.input_model

    def output_schema(self) -> Any:
        """Return the declared output schema/type, if any."""

        return self.output_type or self.output_model

    def validate_input(self, task: Any) -> Any:
        """Validate/coerce a task using the tactic's Pydantic schema."""

        schema = self.input_schema()
        if schema is None:
            return task
        return TypeAdapter(schema).validate_python(task)

    def validate_output(self, result: Any) -> Any:
        """Validate/coerce a result using the tactic's Pydantic schema."""

        schema = self.output_schema()
        if schema is None:
            return result
        return TypeAdapter(schema).validate_python(result)

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
        """Compatibility name used by tactic-tool adapters."""

        return self.name or type(self).__name__

    def run(self, input_value: Any, **kwargs) -> Any:
        """Compatibility alias for the protocol sync tactic surface."""

        return self.__call__(input_value, **kwargs)

    async def arun(self, input_value: Any, **kwargs) -> Any:
        """Compatibility alias for the protocol async tactic surface."""

        return await self.acall(input_value, **kwargs)

    def info(self) -> Any:
        """Return a protocol-compatible tactic info record."""

        from ....protocol import TacticInfo
        from ....protocol.schema import export_json_schema

        return TacticInfo(
            name=self.tactic_name,
            description=getattr(self, "description", None)
            or inspect.getdoc(type(self))
            or "",
            input_schema=export_json_schema(self.input_schema()),
            output_schema=export_json_schema(self.output_schema()),
            capabilities=tuple(sorted(self.capabilities())),
            runtime_kind="native",
        )

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

    @abstractmethod
    def call(self, task: Any, **kwargs) -> Any:
        pass

    def stream(self, task: Any, **kwargs) -> Any:
        """Return a sync stream for runtimes that support it."""

        raise NotImplementedError(f"{type(self).__name__} does not support stream().")

    async def astream(self, task: Any, **kwargs) -> Any:
        """Return an async stream for runtimes that support it."""

        raise NotImplementedError(f"{type(self).__name__} does not support astream().")

    def events(self, task: Any, **kwargs) -> Any:
        """Return a sync event stream for runtimes that support it."""

        raise NotImplementedError(f"{type(self).__name__} does not support events().")

    async def aevents(self, task: Any, **kwargs) -> Any:
        """Return an async event stream for runtimes that support it."""

        raise NotImplementedError(f"{type(self).__name__} does not support aevents().")

    def _copy_for_call(self) -> "Tactic":
        ctx = copy.copy(self)
        ctx._sub_tactics = dict(self._sub_tactics)
        return ctx

    def _prepare_session(self, session: TacticCallSession) -> None:
        """Runtime-specific hook run on the per-call tactic copy."""

        return None

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
        **kwargs,
    ) -> Any:
        tactic_name = self.name or type(self).__name__
        if session_name is None:
            task_str = _task_to_string(task)
            task_hash = hashlib.md5(task_str.encode()).hexdigest()[:8]
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"{tactic_name}_{task_hash}_{timestamp}"

        ctx = self._copy_for_call()
        tactic_path = self._resolve_tactic_path(tactic_name)

        session = TacticCallSession(
            tactic_name=tactic_name, tactic_path=tactic_path
        )
        session.state = "running"
        ctx._session = session
        ctx._prepare_session(session)
        observer = ctx.context.observer

        logger.info(
            "Tactic '%s' started — session_name=%s", tactic_name, session_name
        )
        with _observe_tactic_call(
            observer,
            tactic=ctx,
            session=session,
            task=task,
            session_name=session_name,
            tags=tags,
            metadata=metadata,
        ) as observer_span:
            try:
                validated_task = ctx.validate_input(task)
                result = ctx.call(validated_task, **kwargs)
                result = ctx.validate_output(result)
                session.success(result)
                _notify_tactic_observer(
                    observer,
                    "success",
                    tactic=ctx,
                    session=session,
                    task=validated_task,
                    result=result,
                    span=observer_span,
                    tags=tags,
                    metadata=metadata,
                )
                logger.info(
                    "Tactic '%s' completed — cost=%s agent_calls=%d",
                    tactic_name,
                    session.total_cost.cost,
                    session.agent_call_count,
                )
            except Exception as e:
                session.failure(e)
                _notify_tactic_observer(
                    observer,
                    "failure",
                    tactic=ctx,
                    session=session,
                    task=task,
                    exception=e,
                    span=observer_span,
                    tags=tags,
                    metadata=metadata,
                )
                logger.error(
                    "Tactic '%s' failed: %s",
                    tactic_name,
                    e,
                    exc_info=True,
                )
                raise
        return session if return_session else result

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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
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
                loop.run_in_executor(pool, _run, i, t) for i, t in enumerate(tasks)
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
        """Compatibility shortcut for LLLM's native single-agent runtime."""

        from ..native.tactic import NativeTactic

        return NativeTactic.quick(
            query=query,
            system_prompt=system_prompt,
            model=model,
            return_agent=return_agent,
            **model_args,
        )

    def __repr__(self) -> str:
        parts = [f"{type(self).__name__}(name={self.name!r}"]
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
