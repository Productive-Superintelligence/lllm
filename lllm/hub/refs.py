"""Hub resource refs; the parser lives in :mod:`lllm.protocol.refs`."""

from __future__ import annotations

from ..protocol.refs import REF_SECTIONS, ResourceRef, parse_resource_ref


def validate_resource_ref(ref: str) -> None:
    parse_resource_ref(ref)


__all__ = ["REF_SECTIONS", "ResourceRef", "parse_resource_ref", "validate_resource_ref"]
