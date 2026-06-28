"""LLLM: protocol-first service infrastructure for agentic tactics."""

from .protocol import (
    CallContext,
    CallResult,
    CallTrace,
    LLLMError,
    ProtocolError,
    SchemaError,
    Tactic,
    TacticError,
    TacticEvent,
    TacticInfo,
    TacticLoadError,
    TacticRef,
    TacticRefError,
    TacticServiceError,
    TacticUnsupportedError,
)
from .create import create_project
from .runtimes import CallableTactic, NativeTacticAdapter, PydanticAITactic, as_tactic, tactic_as_tool
from .services import (
    EndpointSpec,
    RemoteTactic,
    TacticResolver,
    create_service_app,
    create_tactic_app,
    endpoint,
)

__version__ = "0.1.0"

__all__ = [
    "CallContext",
    "CallResult",
    "CallTrace",
    "CallableTactic",
    "EndpointSpec",
    "LLLMError",
    "NativeTacticAdapter",
    "ProtocolError",
    "PydanticAITactic",
    "RemoteTactic",
    "SchemaError",
    "Tactic",
    "TacticError",
    "TacticEvent",
    "TacticInfo",
    "TacticLoadError",
    "TacticRef",
    "TacticRefError",
    "TacticResolver",
    "TacticServiceError",
    "TacticUnsupportedError",
    "__version__",
    "as_tactic",
    "create_project",
    "create_service_app",
    "create_tactic_app",
    "endpoint",
    "tactic_as_tool",
]
