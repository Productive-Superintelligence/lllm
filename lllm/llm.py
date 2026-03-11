"""
Compatibility helpers that historically lived under ``lllm.llm``.

Newer code should import directly from ``lllm.core`` modules, but templates in the wild
still expect ``Prompts`` and the re-exported agent symbols to exist here. Keeping this
module lightweight avoids forcing users to touch legacy scaffolds when upgrading.
"""

from __future__ import annotations

from typing import Optional

from lllm.core.agent import Agent, Orchestrator, build_agent, register_agent_class
from lllm.core.models import PROMPT_REGISTRY, Prompt, register_prompt
from lllm.core.discovery import auto_discover_if_enabled


class Prompts:
    """Resolve prompt objects by path with an optional namespace prefix."""

    def __init__(self, root: Optional[str] = None, *, auto_discover: Optional[bool] = None):
        """
        Args:
            root: Optional namespace prefix that is automatically prepended when
                looking up prompt paths (e.g. ``Prompts("system")("task")`` loads
                ``system/task``).
        """
        self.root = (root or "").strip("/ ")
        self._auto_discover_flag = auto_discover

    def _resolve_path(self, name: str) -> str:
        name = name.strip("/ ")
        if not name:
            raise ValueError("Prompt path cannot be empty")
        if self.root:
            return f"{self.root}/{name}".strip("/")
        return name

    def __call__(self, name: str) -> Prompt:
        auto_discover_if_enabled(self._auto_discover_flag)
        """
        Returns:
            Prompt: The registered prompt matching ``<root>/<name>``.

        Raises:
            KeyError: If the prompt has not been registered yet.
        """
        key = self._resolve_path(name)
        if key not in PROMPT_REGISTRY:
            raise KeyError(
                f"Prompt '{key}' not found in registry. Registered: {list(PROMPT_REGISTRY)}"
            )
        return PROMPT_REGISTRY[key]

    def get(self, name: str, default: Optional[Prompt] = None) -> Optional[Prompt]:
        """Like ``dict.get`` – return the prompt or ``default`` if missing."""
        auto_discover_if_enabled(self._auto_discover_flag)
        key = self._resolve_path(name)
        return PROMPT_REGISTRY.get(key, default)


__all__ = [
    "Agent",
    "Orchestrator",
    "Prompts",
    "Prompt",
    "build_agent",
    "register_agent_class",
    "register_prompt",
]
