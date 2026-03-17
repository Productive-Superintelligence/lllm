"""
Tactics — the top-level abstraction in LLLM.

A Tactic is a local, functional unit of agentic behavior that defines HOW
a group of agents solve a task — the "program" that wires callers (agents)
to functions (prompts).

Core capabilities:
    1. Agent initialization from declarative config (agent_group → AgentSpec → Agent).
    2. Standardized I/O — ``call()`` accepts and returns ``str | BaseModel``.
    3. Sub-tactic composition — nest tactics like ``nn.Module`` child modules.
    4. Transparent session tracking via ``_TrackedAgent`` proxy.
    5. Per-call agent isolation — concurrent calls are thread-safe.
    6. Sync batch / async / concurrent execution via thread pool.
    7. Quick constructor — single-agent usage without config or discovery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import hashlib
import datetime as dt
import asyncio
import concurrent.futures
import traceback as tb
import warnings
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import logging

from lllm.core.agent import Agent, AgentCallSession
from lllm.core.prompt import Prompt, InvokeCost
from lllm.core.dialog import Message
from lllm.core.runtime import Runtime, get_default_runtime
from lllm.core.const import APITypes
from lllm.core.config import AgentSpec, parse_agent_configs
from lllm.logging import LogStore
from lllm.invokers import build_invoker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _TrackedAgent — transparent session recording proxy
# ---------------------------------------------------------------------------


class _TrackedAgent:
    """
    Thin proxy around Agent that intercepts ``respond()`` to record
    the ``AgentCallSession`` into the tactic's session.

    All other Agent methods delegate transparently via ``__getattr__``.
    """

    __slots__ = ("_agent", "_session", "_name")

    def __init__(self, agent: Agent, session: TacticCallSession, name: str):
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_name", name)

    def respond(
        self,
        alias: str = None,
        metadata: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        parser_args: Optional[Dict[str, Any]] = None,
        return_session: bool = False,
    ) -> Union[Message, AgentCallSession]:
        agent_session = self._agent.respond(
            alias=alias,
            metadata=metadata,
            args=args,
            parser_args=parser_args,
            return_session=True,
        )
        self._session.record_agent_call(self._name, agent_session)

        if return_session:
            return agent_session
        return agent_session.delivery

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _TrackedAgent.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._agent, name, value)

    def __repr__(self) -> str:
        return f"_TrackedAgent({self._name!r}, agent={self._agent!r})"


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
    tactic_path: Optional[str] = None  # stable ID: "{package_name}::{tactic_name}", e.g. "my_pkg::researcher"

    state: str = "initial"

    agent_sessions: Dict[str, List[AgentCallSession]] = Field(default_factory=dict)
    sub_tactic_sessions: Dict[str, List["TacticCallSession"]] = Field(default_factory=dict)

    delivery: Optional[Any] = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None  # full traceback when state == "failure"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def record_agent_call(self, agent_name: str, session: AgentCallSession) -> None:
        if agent_name not in self.agent_sessions:
            self.agent_sessions[agent_name] = []
        self.agent_sessions[agent_name].append(session)

    def record_sub_tactic_call(self, tactic_name: str, session: "TacticCallSession") -> None:
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
    def agent_cost(self) -> InvokeCost:
        total = InvokeCost()
        for sessions in self.agent_sessions.values():
            for s in sessions:
                total = total + s.cost
        return total

    @property
    def sub_tactic_cost(self) -> InvokeCost:
        total = InvokeCost()
        for sessions in self.sub_tactic_sessions.values():
            for s in sessions:
                total = total + s.total_cost
        return total

    @property
    def total_cost(self) -> InvokeCost:
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
# Registration helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: Any) -> str:
    if isinstance(name, Enum) or (isinstance(name, type) and issubclass(name, Enum)):
        return name.value
    elif isinstance(name, str):
        return name
    else:
        raise ValueError(f"Invalid tactic name: {name}")


def register_tactic_class(tactic_cls, runtime=None):
    runtime = runtime or get_default_runtime()
    name = _normalize_name(getattr(tactic_cls, "name", None))
    assert name not in (None, ""), (
        f"Tactic class {tactic_cls.__name__} must define `name`"
    )
    runtime.register_tactic(name, tactic_cls, overwrite=True)
    return tactic_cls

def get_tactic_class(name, runtime=None):
    runtime = runtime or get_default_runtime()
    return runtime.get_tactic(name)


def _stable_tactic_id(namespace: str, tactic_name: str) -> str:
    """Return the stable physical identifier for a tactic: ``pkg::name``.

    The identifier is derived from the package name (first component of the
    node namespace, e.g. ``"my_pkg"`` from ``"my_pkg.tactics"``) and the
    tactic's ``name`` class attribute.

    It is intentionally independent of:
    - file/folder structure inside the package
    - ``under`` prefix in lllm.toml
    - aliases (``as`` keyword in dependencies)

    Only two semantic changes break it:
    - renaming ``[package] name`` in lllm.toml
    - renaming ``Tactic.name``

    When the tactic is used outside the package system (no namespace),
    falls back to the bare ``tactic_name``.
    """
    if namespace:
        # "my_pkg.tactics" → "my_pkg", "my_pkg" → "my_pkg"
        package_name = namespace.split(".")[0]
        return f"{package_name}::{tactic_name}"
    return tactic_name


def build_tactic(
    config: Dict[str, Any],
    ckpt_dir: str,
    log_store: Optional[LogStore] = None,
    name: str = None,
    runtime: Runtime = None,
    **kwargs,
) -> "Tactic":
    """Build a Tactic from a config dict.

    If *name* is not provided, reads ``config["tactic_type"]``.
    If the config has a ``base`` key, it should already be resolved
    before calling this (use ``resolve_config`` first).
    """
    if name is None:
        name = config.get("tactic_type")
    name = _normalize_name(name)
    rt = runtime or get_default_runtime()
    tactic_cls = get_tactic_class(name, rt)

    # Build the stable tactic ID: "{package_name}::{tactic_name}".
    # This is independent of file layout, "under" prefixes, and aliases —
    # only a package rename or Tactic.name change would alter it.
    try:
        node = rt.get_node(name, resource_type="tactic")
        tactic_path = _stable_tactic_id(node.namespace, tactic_cls.name)
    except (KeyError, AttributeError):
        tactic_path = tactic_cls.name  # not in package system — bare name

    return tactic_cls(
        config, ckpt_dir,
        log_store=log_store, runtime=rt,
        tactic_path=tactic_path,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tactic
# ---------------------------------------------------------------------------


class Tactic(ABC):
    """
    A Tactic is a local, functional unit of agentic behavior.

    It defines HOW a group of agents solve a task — the "program" that
    wires callers (agents) to functions (prompts).

    **Config format** (the dict passed to ``__init__``)::

        tactic_type: analytica
        global:
            model_name: gpt-4o
            model_args:
                temperature: 0.1
        agent_configs:
            - name: analyzer
              system_prompt_path: analytica/analyzer_system
              model_args:
                  max_completion_tokens: 20000
            - name: synthesizer
              system_prompt_path: analytica/synthesizer_system

    ``global`` provides defaults merged into each agent config.
    ``agent_configs`` is a list; each entry must have a ``name``.
    """

    name: str = None
    agent_group: List[str] = None

    # -- Auto-registration ------------------------------------------------

    def __init_subclass__(cls, register: bool = True, runtime: Optional[Runtime] = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if register and getattr(cls, "name", None):
            register_tactic_class(cls, runtime=runtime or get_default_runtime())

    # -- Construction -----------------------------------------------------

    def __init__(
        self,
        config: Dict[str, Any],
        ckpt_dir: str,
        log_store: Optional[LogStore] = None,
        runtime: Optional[Runtime] = None,
        tactic_path: Optional[str] = None,
    ):
        self._runtime = runtime or get_default_runtime()
        self._sub_tactics: Dict[str, Tactic] = {}

        self.config = config
        self.ckpt_dir = ckpt_dir
        self._log_store: Optional[LogStore] = log_store
        self._log_store_warned: bool = False
        # Absolute qualified key in the runtime, e.g. "my_pkg.tactics:folder/researcher".
        # If not supplied by build_tactic, resolved lazily on first use.
        self._tactic_path: Optional[str] = tactic_path
        self.llm_invoker = build_invoker(config)

        # Parse agent specs from the new config format
        assert self.agent_group is not None, (
            f"agent_group not set for tactic '{self.name}'"
        )
        self._agent_specs = parse_agent_configs(
            config, self.agent_group, self.name
        )

        self._max_workers: int = config.get("max_workers", 4)

        # Per-call state — set by _execute on the copy, created here for convenience and checking
        self.agents: Dict[str, Union[Agent, _TrackedAgent]] = self._create_fresh_agents()
        self._session: Optional[TacticCallSession] = None

    # -- Sub-tactic composition -------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Tactic) and name not in ("_sub_tactics",):
            if hasattr(self, "_sub_tactics"):
                self._sub_tactics[name] = value
        super().__setattr__(name, value)

    @property
    def sub_tactics(self) -> Dict[str, "Tactic"]:
        return dict(self._sub_tactics)

    # -- Fresh agent creation (per-call) ----------------------------------

    def _create_fresh_agents(self) -> Dict[str, Agent]:
        return {
            agent_name: spec.build(self._runtime, self.llm_invoker)
            for agent_name, spec in self._agent_specs.items()
        }

    # -- Core execution ---------------------------------------------------

    @abstractmethod
    def call(self, task: Union[str, BaseModel], **kwargs) -> Union[str, BaseModel]:
        pass

    def _execute(
        self,
        task: Union[str, BaseModel],
        session_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        return_session: bool = False,
        **kwargs,
    ) -> Union[str, BaseModel, TacticCallSession]:
        if session_name is None:
            task_str = task if isinstance(task, str) else task.model_dump_json()
            task_hash = hashlib.md5(task_str.encode()).hexdigest()[:8]
            session_name = (
                f"{self.name}_{task_hash}"
                f"_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        ctx = copy.copy(self)
        ctx._sub_tactics = dict(self._sub_tactics)

        # Resolve stable tactic ID lazily (once, then cached).
        # Format: "{package_name}::{tactic_name}", e.g. "my_pkg::researcher".
        # Independent of file layout, "under" prefixes, and aliases.
        if self._tactic_path is None:
            try:
                node = self._runtime.get_node(self.name, resource_type="tactic")
                self._tactic_path = _stable_tactic_id(node.namespace, self.name)
            except (KeyError, AttributeError):
                self._tactic_path = self.name  # not in package system

        session = TacticCallSession(tactic_name=self.name, tactic_path=self._tactic_path)
        session.state = "running"
        ctx._session = session

        raw_agents = ctx._create_fresh_agents()
        ctx.agents = {
            n: _TrackedAgent(agent, session, n)
            for n, agent in raw_agents.items()
        }

        logger.info("Tactic '%s' started — session_name=%s", self.name, session_name)
        try:
            result = ctx.call(task, **kwargs)
            session.success(result)
            logger.info(
                "Tactic '%s' completed — cost=%s agent_calls=%d",
                self.name,
                session.total_cost.cost,
                session.agent_call_count,
            )
        except Exception as e:
            session.failure(e)
            logger.error(
                "Tactic '%s' failed: %s",
                self.name, e, exc_info=True,
            )
            raise
        finally:
            if self._log_store is not None:
                try:
                    saved_id = self._log_store.save_session(
                        session, tags=tags, metadata=metadata
                    )
                    logger.debug("Session persisted — id=%s", saved_id)
                except Exception:
                    logger.warning(
                        "LogStore failed to save session for tactic '%s'",
                        self.name, exc_info=True,
                    )
            elif not self._log_store_warned:
                self._log_store_warned = True
                warnings.warn(
                    f"No LogStore configured for tactic '{self.name}'. "
                    "Session data will not be persisted. "
                    "Pass a LogStore instance via the log_store parameter.",
                    UserWarning,
                    stacklevel=3,
                )

        return session if return_session else result

    def __call__(self, task, session_name=None, tags=None, metadata=None,  return_session=False, **kwargs):
        return self._execute(task, session_name, tags=tags, metadata=metadata, return_session=return_session, **kwargs)

    async def acall(self, task, tags=None, metadata=None, return_session=False, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._execute(task, tags=tags, metadata=metadata, return_session=return_session, **kwargs),
        )

    def bcall(self, tasks, max_workers=None, fail_fast=True, tags=None, metadata=None, return_sessions=False, **kwargs):
        workers = max_workers or self._max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._execute, t, None, tags, metadata, return_session=return_sessions, **kwargs)
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

    async def ccall(self, tasks, max_workers=None, tags=None, metadata=None, return_sessions=False, **kwargs):
        workers = max_workers or self._max_workers
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            def _run(idx, t):
                return idx, self._execute(t, tags=tags, metadata=metadata, return_session=return_sessions, **kwargs)
            futures = [loop.run_in_executor(pool, _run, i, t) for i, t in enumerate(tasks)]
            for coro in asyncio.as_completed(futures):
                idx, result = await coro
                yield idx, result

    # -- Quick constructor ------------------------------------------------

    @classmethod
    def quick(cls, system_prompt: Union[str, Prompt], model: str = "gpt-4o", **model_args) -> Agent:
        if isinstance(system_prompt, str):
            prompt = Prompt(path="_quick/system", prompt=system_prompt)
        else:
            prompt = system_prompt
        invoker = build_invoker({"invoker": "litellm"})
        return Agent(
            name="assistant",
            system_prompt=prompt,
            model=model,
            llm_invoker=invoker,
            model_args=model_args,
        )

    def __repr__(self) -> str:
        parts = [f"Tactic(name={self.name!r}"]
        parts.append(f"agents={list(self._agent_specs.keys())}")
        if self._sub_tactics:
            parts.append(f"sub_tactics={list(self._sub_tactics.keys())}")
        return ", ".join(parts) + ")"
