"""Runtime adapters."""

from .native import NativeTacticAdapter
from .pydantic_ai import PydanticAITactic, PydanticAITacticConfig, tactic_as_tool
from .python import CallableTactic, as_tactic

__all__ = [
    "CallableTactic",
    "NativeTacticAdapter",
    "PydanticAITactic",
    "PydanticAITacticConfig",
    "as_tactic",
    "tactic_as_tool",
]
