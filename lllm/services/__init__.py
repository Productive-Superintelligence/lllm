"""Service adapters and endpoint metadata."""

from .client import RemoteTactic, RemoteTacticError
from .endpoints import EndpointSpec, endpoint
from .fastapi import ErrorDetail, ErrorResponse, RunRequest, RunResponse, create_service_app, create_tactic_app
from .resolver import TacticResolver

__all__ = [
    "EndpointSpec",
    "ErrorDetail",
    "ErrorResponse",
    "RunRequest",
    "RunResponse",
    "RemoteTactic",
    "RemoteTacticError",
    "TacticResolver",
    "create_service_app",
    "create_tactic_app",
    "endpoint",
]
