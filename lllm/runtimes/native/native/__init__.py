"""LLLM native Agent/Prompt/Dialog runtime.

The native runtime is useful for transparent, hackable agent orchestration.
Protocol and service users can depend on ``lllm.protocol`` without importing
these native runtime pieces.
"""

from ..core.agent import Agent
from ..core.const import APITypes, APIType, Modalities, Modality, ParseError, Role, Roles
from ..core.dialog import (
    ContextManager,
    DefaultContextManager,
    Dialog,
    DialogTreeNode,
    Message,
    TokenLogprob,
)
from ..core.native_config import (
    AgentSpec,
    ContextManagerConfig,
    ProxyConfig,
    SkillsConfig,
    parse_agent_configs,
)
from ..core.prompt import (
    AgentCallSession,
    AgentException,
    BaseHandler,
    BaseParser,
    BaseRenderer,
    DefaultSimpleHandler,
    DefaultTagParser,
    Function,
    FunctionCall,
    MCP,
    Prompt,
    StringFormatterRenderer,
    register_prompt,
    tactic_as_function,
    tool,
)
from ..core.resource import load_prompt, load_proxy, load_tool
from ..core.tactic_tool import tactictool
from .tactic import NativeTactic, _TrackedAgent

__all__ = [
    "Agent",
    "AgentCallSession",
    "AgentException",
    "AgentSpec",
    "APITypes",
    "APIType",
    "BaseHandler",
    "BaseParser",
    "BaseRenderer",
    "ContextManager",
    "ContextManagerConfig",
    "DefaultContextManager",
    "DefaultSimpleHandler",
    "DefaultTagParser",
    "Dialog",
    "DialogTreeNode",
    "Function",
    "FunctionCall",
    "MCP",
    "Message",
    "Modalities",
    "Modality",
    "NativeTactic",
    "ParseError",
    "Prompt",
    "ProxyConfig",
    "Role",
    "Roles",
    "SkillsConfig",
    "StringFormatterRenderer",
    "TokenLogprob",
    "_TrackedAgent",
    "load_prompt",
    "load_proxy",
    "load_tool",
    "parse_agent_configs",
    "register_prompt",
    "tactic_as_function",
    "tactictool",
    "tool",
]
