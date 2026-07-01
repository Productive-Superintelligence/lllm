"""Small helpers for PsiHub integration.

PsiHub owns package manifests and validation. LLLM only exposes tactic metadata
in a shape PsiHub can consume.
"""

from __future__ import annotations

from typing import Any

from ..protocol import Tactic
from ..protocol._validation import copy_boundary_value, public_boundary_value
from ..protocol.refs import optional_service_ref_value, optional_tactic_ref_value
from ..services.endpoints import custom_endpoints


def tactic_resource(tactic: Tactic[Any, Any]) -> dict[str, Any]:
    """Return manifest-friendly metadata for a tactic."""

    info = tactic.info()
    package_ref = optional_tactic_ref_value(info.package_ref, "package_ref")
    service_ref = optional_service_ref_value(info.service_ref, "service_ref")
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
        "input_schema": copy_boundary_value(info.input_schema),
        "output_schema": copy_boundary_value(info.output_schema),
        "package_ref": package_ref,
        "service_ref": service_ref,
        "examples": public_boundary_value(info.examples),
        "metadata": public_boundary_value(info.metadata),
    }
