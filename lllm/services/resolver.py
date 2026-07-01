"""Local tactic ref resolver."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..protocol import CallContext, Tactic, TacticRef, TacticRefError
from ..protocol._validation import (
    copy_boundary_value,
    is_sensitive_metadata_key,
    metadata_mapping_value,
)
from .client import RemoteTactic


_NON_TACTIC_CONFIG_REF_SECTIONS = {
    "schemas",
    "services",
    "channels",
    "snapshots",
    "runs",
    "configs",
    "docs",
    "examples",
    "assets",
}


class TacticResolver:
    """Resolve tactic refs to in-process tactics or HTTP clients."""

    def __init__(self) -> None:
        self._bindings: dict[str, Tactic[Any, Any]] = {}

    @classmethod
    def from_config(cls, path: str | Path) -> "TacticResolver":
        """Load URL bindings from a local `.psi/config.toml` style file."""

        resolver = cls()
        config = _load_toml(path)
        refs = config.get("refs", {})
        if not isinstance(refs, dict):
            raise TacticRefError("[refs] must be a TOML table.")
        for raw_ref, data in refs.items():
            if not isinstance(data, dict):
                raise TacticRefError(f"Ref binding must be a table: {raw_ref}")
            is_tactic_ref = _is_tactic_config_ref(raw_ref)
            if is_tactic_ref:
                _validate_tactic_url_binding_targets(raw_ref, data)
            if "url" not in data:
                continue
            if not is_tactic_ref:
                continue
            url = data["url"]
            if not isinstance(url, str) or not url.strip():
                raise TacticRefError(
                    f"Tactic URL binding must be a non-empty string: {raw_ref}"
                )
            if any(ch.isspace() for ch in url):
                raise TacticRefError(
                    f"Tactic URL binding must not contain whitespace: {raw_ref}"
                )
            try:
                resolver.bind_url(
                    raw_ref,
                    url,
                    metadata=_config_ref_metadata(raw_ref, data),
                )
            except (TypeError, ValueError) as exc:
                raise TacticRefError(
                    f"Tactic URL binding is invalid for {raw_ref}: {exc}"
                ) from exc
        return resolver

    def register(self, ref: str | TacticRef, tactic: Tactic[Any, Any]) -> None:
        parsed = TacticRef.parse(ref)
        tactic = _require_tactic(tactic)
        tactic.package_ref = str(parsed)
        self._bindings[str(parsed)] = tactic

    def bind_url(
        self,
        ref: str | TacticRef,
        url: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        **remote_kwargs: Any,
    ) -> RemoteTactic:
        parsed = TacticRef.parse(ref)
        metadata_value = _metadata_mapping(metadata, str(parsed))
        metadata_value["ref"] = str(parsed)
        try:
            remote = RemoteTactic(
                url,
                name=parsed.name,
                metadata=metadata_value,
                **remote_kwargs,
            )
        except (TypeError, ValueError) as exc:
            raise TacticRefError(
                f"Tactic URL binding is invalid for {parsed}: {exc}"
            ) from exc
        remote.package_ref = str(parsed)
        self._bindings[str(parsed)] = remote
        return remote

    def resolve(self, ref: str | TacticRef) -> Tactic[Any, Any]:
        parsed = TacticRef.parse(ref)
        try:
            return self._bindings[str(parsed)]
        except KeyError as exc:
            raise TacticRefError(f"Tactic ref is not bound: {parsed}") from exc

    def run(
        self,
        ref: str | TacticRef,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.resolve(ref).run(input_value, context=context, **kwargs)

    async def arun(
        self,
        ref: str | TacticRef,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self.resolve(ref).arun(input_value, context=context, **kwargs)

    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(self._bindings))


def _load_toml(path: str | Path) -> dict[str, Any]:
    target = Path(_path_value(path, "config path"))
    if target.is_dir():
        target = target / ".psi" / "config.toml"
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib  # type: ignore[no-redef]
    with target.open("rb") as handle:
        return tomllib.load(handle)


def _config_ref_metadata(ref: str, data: dict[str, Any]) -> dict[str, Any]:
    if "metadata" in data:
        metadata = data["metadata"]
        if not isinstance(metadata, dict):
            raise TacticRefError(f'[refs."{ref}".metadata] must be a TOML table.')
    else:
        metadata = {}
    extras = {
        key: value
        for key, value in data.items()
        if key not in {"url", "store", "path", "object", "metadata"}
    }
    return {**extras, **metadata}


def _validate_tactic_url_binding_targets(ref: str, data: dict[str, Any]) -> None:
    targets = [name for name in ("url", "store", "path", "object") if name in data]
    if targets and "url" not in targets:
        targets_text = ", ".join(targets)
        raise TacticRefError(
            "Tactic URL binding must declare a url target, "
            f"got {targets_text}: {ref}"
        )
    if len(targets) > 1:
        targets_text = ", ".join(targets)
        raise TacticRefError(
            "Tactic URL binding must declare only one concrete target, "
            f"got {targets_text}: {ref}"
        )


def _metadata_mapping(value: Any, ref: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TacticRefError(f"Tactic URL binding metadata must be a mapping: {ref}")
    try:
        metadata = metadata_mapping_value("Tactic URL binding metadata", value)
    except TypeError as exc:
        raise TacticRefError(f"{exc}: {ref}") from exc
    _reject_sensitive_metadata(metadata, ref)
    return metadata


def _reject_sensitive_metadata(value: Any, ref: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if is_sensitive_metadata_key(key):
                raise TacticRefError(
                    "Tactic URL binding metadata must not include raw secret "
                    f"key {key!r}: {ref}"
                )
            _reject_sensitive_metadata(item, ref)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_sensitive_metadata(item, ref)

def _is_tactic_config_ref(ref: str) -> bool:
    if not isinstance(ref, str) or not ref.strip():
        raise TacticRefError("Ref binding key must be a non-empty string.")
    parsed = urlparse(ref)
    raw_parts = parsed.path.split("/")
    if (
        parsed.scheme != "psi"
        or not parsed.netloc
        or parsed.params
        or parsed.query
        or parsed.fragment
        or len(raw_parts) != 4
        or raw_parts[0] != ""
    ):
        raise TacticRefError(f"Ref binding key must be a psi:// resource ref: {ref}")
    package, resource_kind, name = raw_parts[1:]
    for segment in (parsed.netloc, package, resource_kind, name):
        decoded = unquote(segment)
        if (
            decoded in {".", ".."}
            or decoded != segment
            or "%" in segment
            or not decoded.strip()
            or any(ch.isspace() for ch in decoded)
            or any(ch in decoded for ch in "/\\:;")
        ):
            raise TacticRefError(
                f"Ref binding key must use plain path segments: {ref}"
            )
    if resource_kind == "tactics":
        return True
    if resource_kind in _NON_TACTIC_CONFIG_REF_SECTIONS:
        return False
    raise TacticRefError(
        f"Ref binding key uses unknown resource section {resource_kind!r}: {ref}"
    )


def _path_value(value: Any, label: str) -> str:
    try:
        text = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a non-empty path string") from exc
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    return text


def _require_tactic(value: Any) -> Tactic[Any, Any]:
    if not isinstance(value, Tactic):
        raise TypeError("tactic must be a Tactic instance.")
    return value
