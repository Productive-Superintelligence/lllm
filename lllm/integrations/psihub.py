"""Small helpers for PsiHub integration.

PsiHub owns package manifests and validation. LLLM only exposes tactic metadata
in a shape PsiHub can consume.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..protocol import Tactic
from ..services.endpoints import custom_endpoints


def tactic_resource(tactic: Tactic[Any, Any]) -> dict[str, Any]:
    """Return manifest-friendly metadata for a tactic."""

    info = tactic.info()
    return {
        "name": info.name,
        "description": info.description,
        "runtime": info.runtime_kind,
        "capabilities": list(info.capabilities),
        "endpoints": [
            {
                "name": spec.name,
                "method": spec.method,
                "path": spec.path,
                "mode": spec.mode,
                "description": spec.description,
                "tags": list(spec.tags),
            }
            for spec, _method in custom_endpoints(tactic)
        ],
        "input_schema": deepcopy(info.input_schema),
        "output_schema": deepcopy(info.output_schema),
        "package_ref": info.package_ref,
        "service_ref": info.service_ref,
        "examples": deepcopy(info.examples),
        "metadata": deepcopy(info.metadata),
    }
