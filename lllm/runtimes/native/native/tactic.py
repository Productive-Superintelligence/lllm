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
                "`pip install 'lllm-core[native]'`."
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
    """A tactic that must declare ``agent_group``.

    Every :class:`~lllm.Tactic` builds native agents when ``agent_group`` is
    set; this subclass only insists on it, for code written against earlier
    releases that subclassed ``NativeTactic``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if self.agent_group is None:
            raise AssertionError(
                f"agent_group not set for native tactic '{self.name}'"
            )
        super().__init__(*args, **kwargs)
