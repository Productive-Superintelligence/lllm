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
