"""Compatibility facade for core LLLM modules.

Concrete submodules remain the best import target for new code. This package
initializer is intentionally lazy so protocol imports do not pull in the native
Agent/Prompt/Dialog runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Config and resources
    "NATIVE_RESOURCE_TYPES": (".resource", "NATIVE_RESOURCE_TYPES"),
    "PLATFORM_RESOURCE_TYPES": (".resource", "PLATFORM_RESOURCE_TYPES"),
    "PackageInfo": (".resource", "PackageInfo"),
    "ResourceNode": (".resource", "ResourceNode"),
    "load_asset": (".resource", "load_asset"),
    "load_config": (".resource", "load_config"),
    "load_prompt": (".resource", "load_prompt"),
    "load_proxy": (".resource", "load_proxy"),
    "load_resource": (".resource", "load_resource"),
    "load_service_ref": (".resource", "load_service_ref"),
    "load_tactic": (".resource", "load_tactic"),
    "load_tool": (".resource", "load_tool"),
    "resource_category": (".resource", "resource_category"),
    "resolve_config": (".config", "resolve_config"),
    # Provider-neutral native primitives.
    "APITypes": (".const", "APITypes"),
    "APIType": (".const", "APIType"),
    "FunctionCall": (".const", "FunctionCall"),
    "InvokeCost": (".const", "InvokeCost"),
    "InvokeResult": (".const", "InvokeResult"),
    "Invokers": (".const", "Invokers"),
    "Modalities": (".const", "Modalities"),
    "Modality": (".const", "Modality"),
    "ParseError": (".const", "ParseError"),
    "Role": (".const", "Role"),
    "Roles": (".const", "Roles"),
    # Registry and tactic protocol
    "Registry": (".runtime", "Registry"),
    "Runtime": (".runtime", "Runtime"),
    "Tactic": (".tactic", "Tactic"),
    "TacticCallSession": (".tactic", "TacticCallSession"),
    "TacticContext": (".tactic", "TacticContext"),
    "build_tactic": (".tactic_registry", "build_tactic"),
    "get_tactic_class": (".tactic_registry", "get_tactic_class"),
    "register_tactic_class": (".tactic_registry", "register_tactic_class"),
    "tactictool": (".tactic_tool", "tactictool"),
    # Native runtime compatibility exports. Prefer lllm.native in new code.
    "Agent": ("..native", "Agent"),
    "AgentSpec": ("..native", "AgentSpec"),
    "ContextManager": ("..native", "ContextManager"),
    "DefaultContextManager": ("..native", "DefaultContextManager"),
    "Dialog": ("..native", "Dialog"),
    "DialogTreeNode": ("..native", "DialogTreeNode"),
    "Message": ("..native", "Message"),
    "NativeTactic": ("..native", "NativeTactic"),
    "Prompt": ("..native", "Prompt"),
    "TokenLogprob": ("..native", "TokenLogprob"),
    "parse_agent_configs": ("..native", "parse_agent_configs"),
    # Prompt/tool compatibility exports.
    "AgentCallSession": (".prompt", "AgentCallSession"),
    "AgentException": (".prompt", "AgentException"),
    "BaseHandler": (".prompt", "BaseHandler"),
    "BaseParser": (".prompt", "BaseParser"),
    "BaseRenderer": (".prompt", "BaseRenderer"),
    "DefaultSimpleHandler": (".prompt", "DefaultSimpleHandler"),
    "DefaultTagParser": (".prompt", "DefaultTagParser"),
    "Function": (".prompt", "Function"),
    "MCP": (".prompt", "MCP"),
    "StringFormatterRenderer": (".prompt", "StringFormatterRenderer"),
    "register_prompt": (".prompt", "register_prompt"),
    "tactic_as_function": (".prompt", "tactic_as_function"),
    "tool": (".prompt", "tool"),
    # Text block helpers.
    "find_md_blocks": ("..utils", "find_md_blocks"),
    "find_xml_blocks": ("..utils", "find_xml_blocks"),
}

__all__ = sorted(_LAZY_ATTRS)


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
