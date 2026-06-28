"""Small helpers for PsiHub integration.

PsiHub owns package manifests and validation. LLLM only exposes tactic metadata
in a shape PsiHub can consume.
"""

from __future__ import annotations

from typing import Any

from ..protocol import Tactic


def tactic_resource(tactic: Tactic[Any, Any]) -> dict[str, Any]:
    """Return manifest-friendly metadata for a tactic."""

    info = tactic.info()
    return {
        "name": info.name,
        "description": info.description,
        "runtime": info.runtime_kind,
        "capabilities": list(info.capabilities),
        "input_schema": info.input_schema,
        "output_schema": info.output_schema,
        "package_ref": info.package_ref,
        "service_ref": info.service_ref,
        "metadata": dict(info.metadata),
    }
