# lllm/core/context.py
from __future__ import annotations
from typing import Dict, Any, Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lllm.core.prompt import Prompt
    from lllm.proxies.base import BaseProxy


class Context:
    """
    Holds all registries and shared state for an LLLM runtime.
    
    Most users never touch this directly — the module-level functions
    (register_prompt, register_proxy, etc.) operate on a default instance.
    Researchers who need isolation (testing, parallel experiments) can
    create their own.
    """

    def __init__(self):
        self.prompts: Dict[str, Prompt] = {}
        self.proxies: Dict[str, Type[BaseProxy]] = {}
        self.agents: Dict[str, Type] = {}  # Orchestrator subclasses
        self._discovery_done: bool = False

    # -- Prompts --
    def register_prompt(self, prompt: Prompt, overwrite: bool = True):
        if prompt.path in self.prompts and not overwrite:
            raise ValueError(f"Prompt '{prompt.path}' already registered")
        self.prompts[prompt.path] = prompt

    def get_prompt(self, path: str) -> Prompt:
        if path not in self.prompts:
            raise KeyError(f"Prompt '{path}' not found. Registered: {sorted(self.prompts)}")
        return self.prompts[path]

    # -- Proxies --
    def register_proxy(self, name: str, proxy_cls: Type[BaseProxy], overwrite: bool = False):
        if name in self.proxies and not overwrite:
            raise ValueError(f"Proxy '{name}' already registered")
        self.proxies[name] = proxy_cls

    # -- Agents --
    def register_agent(self, agent_type: str, agent_cls: Type, overwrite: bool = False):
        if agent_type in self.agents and not overwrite:
            raise ValueError(f"Agent type '{agent_type}' already registered")
        self.agents[agent_type] = agent_cls

    # -- Lifecycle --
    def reset(self):
        """Clear all registries. Primarily for testing."""
        self.prompts.clear()
        self.proxies.clear()
        self.agents.clear()
        self._discovery_done = False


# The default instance — created at import time, no side effects
_default_context = Context()


def get_default_context() -> Context:
    return _default_context


def set_default_context(ctx: Context):
    global _default_context
    _default_context = ctx