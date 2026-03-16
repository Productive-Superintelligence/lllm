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
from dataclasses import dataclass, field
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
from lllm.core.log import build_log_base
from lllm.invokers import build_invoker
import lllm.utils as U

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Config parsing helpers
# ---------------------------------------------------------------------------

def _merge_global_into_agent(
    global_cfg: Dict[str, Any],
    agent_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge global defaults into an agent config.

    Agent-level keys override global keys.  ``model_args`` dicts are
    merged (not replaced), so you can set ``temperature`` globally and
    override ``max_tokens`` per agent.
    """
    from lllm.core.config import _deep_merge
    return _deep_merge(global_cfg, agent_cfg)


def parse_agent_configs(
    config: Dict[str, Any],
    agent_group: List[str],
    tactic_name: str,
) -> Dict[str, AgentSpec]:
    """Parse the ``global`` + ``agent_configs`` sections of a config dict.

    Parameters
    ----------
    config:
        The fully resolved tactic config dict (after ``base`` inheritance).
    agent_group:
        The agent names this tactic requires.
    tactic_name:
        For error messages.

    Returns
    -------
    ``{agent_name: AgentSpec}`` for each name in *agent_group*.
    """
    global_cfg = config.get("global", {})
    raw_list = config.get("agent_configs", [])

    # Build a name → merged-config lookup
    agent_by_name: Dict[str, Dict[str, Any]] = {}
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise TypeError(
                f"Each entry in agent_configs must be a dict, got {type(entry).__name__}"
            )
        name = entry.get("name")
        if name is None:
            raise ValueError(
                f"Agent config entry missing 'name' key: {entry}"
            )
        merged = _merge_global_into_agent(global_cfg, entry)
        agent_by_name[name] = merged

    # Validate that all required agents are present
    specs: Dict[str, AgentSpec] = {}
    for agent_name in agent_group:
        if agent_name not in agent_by_name:
            raise ValueError(
                f"Agent '{agent_name}' required by tactic '{tactic_name}' "
                f"not found in agent_configs. "
                f"Available: {sorted(agent_by_name.keys())}"
            )
        specs[agent_name] = AgentSpec.from_config(
            agent_name, agent_by_name[agent_name]
        )

    return specs


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
    state: str = "initial"

    agent_sessions: Dict[str, List[AgentCallSession]] = Field(default_factory=dict)
    sub_tactic_sessions: Dict[str, List["TacticCallSession"]] = Field(default_factory=dict)

    delivery: Optional[Any] = None
    error: Optional[str] = None

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


def build_tactic(
    config: Dict[str, Any],
    ckpt_dir: str,
    stream=None,
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
    tactic_cls = get_tactic_class(name, runtime)
    return tactic_cls(config, ckpt_dir, stream, runtime=runtime, **kwargs)


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
        stream=None,
        runtime: Optional[Runtime] = None,
    ):
        self._runtime = runtime or get_default_runtime()
        self._sub_tactics: Dict[str, Tactic] = {}

        self.config = config
        self.ckpt_dir = ckpt_dir
        self._stream = stream or U.PrintSystem()
        self._stream_backup = self._stream
        self.st = None
        self._log_base = build_log_base(config)
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
            agent_name: spec.build(self._runtime, self.llm_invoker, self._log_base)
            for agent_name, spec in self._agent_specs.items()
        }

    # -- Core execution ---------------------------------------------------

    @abstractmethod
    def call(self, task: Union[str, BaseModel], **kwargs) -> Union[str, BaseModel]:
        pass

    def _execute(
        self,
        task: Union[str, BaseModel],
        session_name: str = None,
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
        session = TacticCallSession(tactic_name=self.name)
        session.state = "running"
        ctx._session = session

        raw_agents = ctx._create_fresh_agents()
        ctx.agents = {
            n: _TrackedAgent(agent, session, n)
            for n, agent in raw_agents.items()
        }

        ctx.set_st(session_name)
        try:
            result = ctx.call(task, **kwargs)
            session.success(result)
        except Exception as e:
            session.failure(e)
            raise
        finally:
            ctx.restore_st()

        return session if return_session else result

    def __call__(self, task, session_name=None, return_session=False, **kwargs):
        return self._execute(task, session_name, return_session, **kwargs)

    async def acall(self, task, return_session=False, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._execute(task, return_session=return_session, **kwargs),
        )

    def bcall(self, tasks, return_sessions=False, max_workers=None, **kwargs):
        workers = max_workers or self._max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._execute, t, None, return_sessions, **kwargs)
                for t in tasks
            ]
            return [f.result() for f in futures]

    async def ccall(self, tasks, return_sessions=False, max_workers=None, **kwargs):
        workers = max_workers or self._max_workers
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            def _run(idx, t):
                return idx, self._execute(t, return_session=return_sessions, **kwargs)
            futures = [loop.run_in_executor(pool, _run, i, t) for i, t in enumerate(tasks)]
            for coro in asyncio.as_completed(futures):
                idx, result = await coro
                yield idx, result

    # -- Session helpers --------------------------------------------------

    def set_st(self, session_name: str) -> None:
        self.st = U.StreamWrapper(self._stream, self._log_base, session_name)

    def restore_st(self) -> None:
        self.st = None

    def silent(self) -> None:
        self._stream = U.PrintSystem(silent=True)

    def restore(self) -> None:
        self._stream = self._stream_backup

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