"""LLLM: Low-Level Language Models.

A lightweight framework for advanced agentic systems. ``lllm.Tactic`` is the
program: write ``call()``, declare ``agent_group`` for native agents, or use
``Tactic.quick()`` for a single agent with no configuration. Every tactic is
also a typed protocol tactic that can be served, packaged, and composed.
"""

from .protocol import (
    CallContext,
    CallResult,
    CallTrace,
    LLLMError,
    ProtocolError,
    SchemaError,
    TacticError,
    TacticEvent,
    TacticInfo,
    TacticLoadError,
    TacticRef,
    TacticRefError,
    TacticServiceError,
    TacticUnsupportedError,
)
from .runtimes.native.core.tactic import Tactic
from .create import create_project
from .parsers import (
    BaseParser,
    DefaultTagParser,
    ParseError,
    find_md_blocks,
    find_xml_blocks,
)
from .proxy import InMemoryProxyLog, ProxyRecord, ProxyTactic, proxy_tactic
from .runtimes import CallableTactic, NativeTacticAdapter, PydanticAITactic, as_tactic, tactic_as_tool
from .sandbox import (
    SandboxError,
    SandboxLimitError,
    SandboxPolicy,
    SandboxTimeoutError,
    SandboxedTactic,
    sandbox_tactic,
)
from .resolver import Resolver
from .services import (
    EndpointSpec,
    RemoteTactic,
    RemoteTacticError,
    TacticResolver,
    create_service_app,
    create_tactic_app,
    endpoint,
)

__version__ = "0.2.1"

# The native vocabulary, as in the original ``lllm`` API. Resolved lazily so
# ``import lllm`` stays light; ``lllm.runtimes.native`` holds the same objects.
_NATIVE_EXPORTS = {
    **{
        name: "lllm.runtimes.native"
        for name in (
            "Agent",
            "AgentSpec",
            "Dialog",
            "Function",
            "FunctionCall",
            "InferenceHints",
            "MCP",
            "Message",
            "NativeTactic",
            "Prompt",
            "Role",
            "Roles",
            "Runtime",
            "parse_agent_configs",
            "register_prompt",
            "tactictool",
            "tool",
        )
    },
    **{
        name: "lllm.runtimes.native.core.runtime"
        for name in (
            "export_package",
            "get_default_runtime",
            "get_runtime",
            "install_package",
            "list_packages",
            "load_runtime",
            "remove_package",
            "set_default_runtime",
        )
    },
    **{
        name: "lllm.runtimes.native.core.resource"
        for name in (
            "load_config",
            "load_prompt",
            "load_proxy",
            "load_resource",
            "load_tactic",
            "load_tool",
        )
    },
    "load_package": "lllm.runtimes.native.core.config",
    "resolve_config": "lllm.runtimes.native.core.config",
    "build_tactic": "lllm.runtimes.native.core.tactic",
    "register_tactic_class": "lllm.runtimes.native.core.tactic",
    "BaseProxy": "lllm.runtimes.native.proxies",
    "ProxyManager": "lllm.runtimes.native.proxies",
    "ProxyRegistrator": "lllm.runtimes.native.proxies",
    "register_proxy": "lllm.runtimes.native.proxies",
    "register_invoker": "lllm.runtimes.native.invokers",
}


def __getattr__(name: str):
    module_name = _NATIVE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'lllm' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_NATIVE_EXPORTS))

__all__ = [
    "CallContext",
    "CallResult",
    "CallTrace",
    "CallableTactic",
    "BaseParser",
    "DefaultTagParser",
    "EndpointSpec",
    "LLLMError",
    "InMemoryProxyLog",
    "NativeTacticAdapter",
    "ParseError",
    "ProtocolError",
    "PydanticAITactic",
    "ProxyRecord",
    "ProxyTactic",
    "RemoteTactic",
    "RemoteTacticError",
    "SandboxError",
    "SandboxLimitError",
    "SandboxPolicy",
    "SandboxTimeoutError",
    "SandboxedTactic",
    "SchemaError",
    "Tactic",
    "TacticError",
    "TacticEvent",
    "TacticInfo",
    "TacticLoadError",
    "TacticRef",
    "TacticRefError",
    "Resolver",
    "TacticResolver",
    "TacticServiceError",
    "TacticUnsupportedError",
    "__version__",
    "as_tactic",
    "create_project",
    "create_service_app",
    "create_tactic_app",
    "endpoint",
    "find_md_blocks",
    "find_xml_blocks",
    "proxy_tactic",
    "sandbox_tactic",
    "tactic_as_tool",
]
