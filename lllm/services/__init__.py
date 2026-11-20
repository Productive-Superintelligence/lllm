"""Service adapters and endpoint metadata."""

from .client import RemoteTactic, RemoteTacticError
from .endpoints import EndpointSpec, endpoint
from .fastapi import (
    CrossSiteRequestError,
    ErrorDetail,
    ErrorResponse,
    RunRequest,
    RunResponse,
    call_context_from_data,
    create_service_app,
    create_tactic_app,
    status_code_for_error,
)
from .resolver import TacticResolver

__all__ = [
    "CrossSiteRequestError",
    "EndpointSpec",
    "ErrorDetail",
    "ErrorResponse",
    "RunRequest",
    "RunResponse",
    "RemoteTactic",
    "RemoteTacticError",
    "TacticResolver",
    "call_context_from_data",
    "create_service_app",
    "create_tactic_app",
    "endpoint",
    "status_code_for_error",
]
