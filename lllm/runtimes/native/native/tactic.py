"""Native LLLM tactic runtime."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

from ..core.tactic import (
    Tactic,
    TacticCallSession,
    TacticContext,
    _config_as_mapping,
)

if TYPE_CHECKING:
    from ..core.agent import Agent
    from ..core.dialog import Message
    from ..core.prompt import Prompt


def _build_native_invoker(config: Dict[str, Any]) -> Any:
    """Build a native LLLM invoker lazily."""

    from ..invokers import build_invoker

    try:
        return build_invoker(config)
    except ModuleNotFoundError as exc:
        if exc.name == "litellm":
            raise RuntimeError(
                "The native LLLM runtime requires LiteLLM. Install it with "
                "`pip install 'lllm[native]'`."
            ) from exc
        raise


def _get_default_registry() -> Any:
    from ..core.runtime import get_default_registry

    return get_default_registry()


class _TrackedAgent:
    """
    Thin proxy around Agent that intercepts ``respond()`` to record
    the ``AgentCallSession`` into the tactic's session.

    All other Agent methods delegate transparently via ``__getattr__``.
    """

    __slots__ = ("_agent", "_session", "_name")

    def __init__(self, agent: Any, session: TacticCallSession, name: str):
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_name", name)

    def respond(
        self,
        alias: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        parser_args: Optional[Dict[str, Any]] = None,
        return_session: bool = False,
    ) -> Any:
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


class NativeTactic(Tactic):
    """Tactic backed by LLLM's native Agent/Prompt/Dialog runtime.

    Subclass this when you want ``agent_group`` and ``agent_configs`` to build
    live LLLM ``Agent`` objects. Plain ``Tactic`` subclasses can use any
    runtime internally, including Pydantic AI, graph runtimes, robot monitors,
    or ordinary Python code.
    """

    agent_group: Optional[List[str]] = None

    def __init__(
        self,
        config: Optional[Any] = None,
        runtime: Any = None,
        registry: Any = None,
        tactic_path: Optional[str] = None,
        context: Optional[TacticContext] = None,
        observer: Any = None,
    ):
        super().__init__(
            config=config,
            runtime=runtime,
            registry=registry,
            tactic_path=tactic_path,
            context=context,
            observer=observer,
        )
        if self.agent_group is None:
            raise AssertionError(
                f"agent_group not set for native tactic '{self.name}'"
            )
        self._runtime = self._registry or _get_default_registry()
        self._llm_invoker = None
        self._agent_specs = {}
        self.agents: Dict[str, Any] = {}
        self._init_native_runtime()

    @property
    def llm_invoker(self):
        if self._llm_invoker is None:
            config = _config_as_mapping(self.config, "Native tactic config")
            self._llm_invoker = _build_native_invoker(config)
        return self._llm_invoker

    @llm_invoker.setter
    def llm_invoker(self, value):
        self._llm_invoker = value

    def _init_native_runtime(self) -> None:
        from ..core.native_config import parse_agent_configs

        config = _config_as_mapping(self.config, "Native tactic config")
        self._agent_specs = parse_agent_configs(config, self.agent_group, self.name)
        self.agents = self._create_fresh_agents()

    def _create_fresh_agents(self) -> Dict[str, Any]:
        if not self._agent_specs:
            return {}
        return {
            agent_name: spec.build(self._runtime, self.llm_invoker)
            for agent_name, spec in self._agent_specs.items()
        }

    def _prepare_session(self, session: TacticCallSession) -> None:
        raw_agents = self._create_fresh_agents()
        self.agents = {
            name: _TrackedAgent(agent, session, name)
            for name, agent in raw_agents.items()
        }

    @classmethod
    def quick(
        cls,
        query: Optional[str] = None,
        system_prompt: Optional[Union[str, "Prompt"]] = "You are a helpful assistant.",
        model: str = "gpt-4o",
        return_agent: bool = False,
        **model_args: Any,
    ) -> Union["Message", "Agent", Tuple["Message", "Agent"]]:
        """Quick constructor for a single native LLLM agent."""

        from ..core.agent import Agent
        from ..core.prompt import Prompt

        if isinstance(system_prompt, str):
            prompt = Prompt(path="_quick/system", prompt=system_prompt)
        else:
            prompt = system_prompt
        invoker = _build_native_invoker({"invoker": "litellm"})
        agent = Agent(
            name="assistant",
            system_prompt=prompt,
            model=model,
            llm_invoker=invoker,
            runtime=_get_default_registry(),
            model_args=model_args,
        )
        if query is not None:
            agent.open("chat")
            agent.receive(query)
            response = agent.respond()
            if return_agent:
                return response, agent
            return response
        return agent

    def __repr__(self) -> str:
        parts = [f"{type(self).__name__}(name={self.name!r}"]
        parts.append(f"agents={list(self._agent_specs.keys())}")
        if self._sub_tactics:
            parts.append(f"sub_tactics={list(self._sub_tactics.keys())}")
        return ", ".join(parts) + ")"
