"""The port ``lllm serve`` listens on for a package resource.

``lllm serve`` and the hub's config template both use :func:`serve_port`, so
the template's URLs point where the resources are served.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

DEFAULT_PORT = 8000


def declared_port(metadata: Any) -> int | None:
    """The ``port`` a resource declares in its metadata, if it is valid."""

    port = metadata.get("port") if isinstance(metadata, dict) else None
    if isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535:
        return port
    return None


def serve_port(metadata: Any, serving_services: Iterable[Any] = ()) -> int:
    """The resource's own ``metadata.port``, else the first declared port.

    ``serving_services`` holds the metadata of the services that serve the
    resource (for a tactic, those naming it in ``tactic`` or ``tactics``), in
    manifest order. Without a declared port, the port is 8000.
    """

    port = declared_port(metadata)
    if port is not None:
        return port
    for service_metadata in serving_services:
        port = declared_port(service_metadata)
        if port is not None:
            return port
    return DEFAULT_PORT
