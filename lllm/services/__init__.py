"""Service adapters and endpoint metadata."""

from .endpoints import EndpointSpec, endpoint
from .fastapi import ErrorDetail, ErrorResponse, RunRequest, RunResponse, create_service_app, create_tactic_app

__all__ = [
    "EndpointSpec",
    "ErrorDetail",
    "ErrorResponse",
    "RunRequest",
    "RunResponse",
    "create_service_app",
    "create_tactic_app",
    "endpoint",
]
