"""Protocol-level exceptions for LLLM."""

from __future__ import annotations


class LLLMError(RuntimeError):
    """Base error raised by LLLM."""


class ProtocolError(LLLMError):
    """Raised when protocol data is invalid."""


class SchemaError(ProtocolError):
    """Raised when a schema cannot validate or describe data."""


class TacticError(ProtocolError):
    """Raised when a tactic boundary fails."""


class TacticUnsupportedError(TacticError):
    """Raised when a tactic does not support a requested capability."""


class TacticLoadError(TacticError):
    """Raised when a tactic entrypoint cannot be loaded."""


class TacticRefError(TacticError):
    """Raised when a tactic reference cannot be parsed or resolved."""


class TacticServiceError(TacticError):
    """Raised when a remote tactic service returns an error."""
