"""Local tactic ref resolver."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..protocol import CallContext, Tactic, TacticRef, TacticRefError
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
            if "url" not in data:
                continue
            if not _is_tactic_config_ref(raw_ref):
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
                resolver.bind_url(raw_ref, url)
            except ValueError as exc:
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
        **remote_kwargs: Any,
    ) -> RemoteTactic:
        parsed = TacticRef.parse(ref)
        remote = RemoteTactic(
            url,
            name=parsed.name,
            metadata={"ref": str(parsed)},
            **remote_kwargs,
        )
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


def _is_tactic_config_ref(ref: str) -> bool:
    if not isinstance(ref, str) or not ref.strip():
        raise TacticRefError("Ref binding key must be a non-empty string.")
    parsed = urlparse(ref)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) == 3 and parts[1] in _NON_TACTIC_CONFIG_REF_SECTIONS:
        return False
    return True


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
