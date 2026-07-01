"""Preserved native LLLM runtime.

The public architecture is protocol-first, but the original native runtime
is kept here for transparent agent/dialog/prompt machinery, tactic registries,
invokers, proxies, and research workflows that need to inspect the inside of
an agent turn.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .adapter import NativeTacticAdapter
from .core import Runtime, Tactic, TacticCallSession, TacticContext, tactictool

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "APITypes": (".core", "APITypes"),
    "APIType": (".core", "APIType"),
    "Agent": (".core", "Agent"),
    "AgentCallSession": (".core", "AgentCallSession"),
    "AgentException": (".core", "AgentException"),
    "AgentSpec": (".core", "AgentSpec"),
    "BaseHandler": (".core", "BaseHandler"),
    "BaseParser": (".core", "BaseParser"),
    "BaseRenderer": (".core", "BaseRenderer"),
    "DefaultSimpleHandler": (".core", "DefaultSimpleHandler"),
    "DefaultTagParser": (".core", "DefaultTagParser"),
    "Dialog": (".core", "Dialog"),
    "DialogTreeNode": (".core", "DialogTreeNode"),
    "Function": (".core", "Function"),
    "FunctionCall": (".core", "FunctionCall"),
    "InvokeCost": (".core", "InvokeCost"),
    "InvokeResult": (".core", "InvokeResult"),
    "Invokers": (".core", "Invokers"),
    "MCP": (".core", "MCP"),
    "Message": (".core", "Message"),
    "Modalities": (".core", "Modalities"),
    "Modality": (".core", "Modality"),
    "NativeTactic": (".core", "NativeTactic"),
    "ParseError": (".core", "ParseError"),
    "Prompt": (".core", "Prompt"),
    "Registry": (".core", "Registry"),
    "Role": (".core", "Role"),
    "Roles": (".core", "Roles"),
    "StringFormatterRenderer": (".core", "StringFormatterRenderer"),
    "TokenLogprob": (".core", "TokenLogprob"),
    "find_md_blocks": (".core", "find_md_blocks"),
    "find_xml_blocks": (".core", "find_xml_blocks"),
    "parse_agent_configs": (".core", "parse_agent_configs"),
    "register_prompt": (".core", "register_prompt"),
    "tactic_as_function": (".core", "tactic_as_function"),
    "tool": (".core", "tool"),
}

__all__ = sorted(
    {
        "NativeTacticAdapter",
        "Runtime",
        "Tactic",
        "TacticCallSession",
        "TacticContext",
        "tactictool",
        *_LAZY_ATTRS,
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
