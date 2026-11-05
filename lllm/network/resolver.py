"""Local channel ref resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .client import AsyncRemoteStore, RemoteStore
from .core import NetworkError
from ..protocol._validation import copy_boundary_value
from ..protocol.refs import parse_resource_ref
from .stores import LocalStore

_CHANNEL_CONFIG_REF_SECTIONS = {"channels", "snapshots"}
_NON_CHANNEL_CONFIG_REF_SECTIONS = {
    "assets",
    "configs",
    "docs",
    "examples",
    "runs",
    "schemas",
    "services",
    "tactics",
}


class ChannelRefError(NetworkError, ValueError):
    """Raised when an LLLM network `lllm://` ref or binding is invalid."""


@dataclass(frozen=True)
class ChannelRef:
    """A stable `lllm://org/package/channels/name` or snapshots ref."""

    value: str
    org: str = field(init=False)
    package: str = field(init=False)
    resource_kind: str = field(init=False)
    name: str = field(init=False)

    def __post_init__(self) -> None:
        ref = parse_resource_ref(
            self.value,
            noun="Channel ref",
            sections=("channels", "snapshots"),
            error=ChannelRefError,
        )
        object.__setattr__(self, "org", ref.org)
        object.__setattr__(self, "package", ref.package)
        object.__setattr__(self, "resource_kind", ref.resource_kind)
        object.__setattr__(self, "name", ref.name)

    @classmethod
    def parse(cls, value: str | "ChannelRef") -> "ChannelRef":
        if isinstance(value, cls):
            return value
        return cls(value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ResolvedChannelRef:
    """A concrete local binding for a channel ref."""

    ref: str
    resource_kind: str
    name: str
    url: str | None = None
    store: str | None = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ChannelResolver:
    """Resolve channel refs to local stores or HTTP clients."""

    def __init__(self, *, root: str | Path = ".") -> None:
        self.root = Path(_path_value(root, "config root")).expanduser().resolve()
        self._bindings: dict[str, ResolvedChannelRef] = {}
        self._stores: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_config(cls, path: str | Path) -> "ChannelResolver":
        root, target = _config_target(path)
        resolver = cls(root=root)
        resolver._load_config(_load_toml(target))
        return resolver

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        root: str | Path | None = None,
    ) -> "ChannelResolver":
        """Parse config text in memory; relative paths resolve against ``root``."""
        text_value = _config_text_value(text)
        resolver = cls(root="." if root is None else root)
        resolver._load_config(_parse_toml(text_value))
        return resolver

    def _load_config(self, config: dict[str, Any]) -> None:
        refs = config.get("refs", {})
        if not isinstance(refs, dict):
            raise ChannelRefError("[refs] must be a TOML table.")
        self._stores = _table_of_tables(config.get("stores", {}), "stores")
        for raw_ref, data in refs.items():
            if not _is_channel_config_ref(raw_ref):
                continue
            if not isinstance(data, dict):
                raise ChannelRefError(f"Ref binding must be a table: {raw_ref}")
            if "metadata" in data and not isinstance(data["metadata"], dict):
                raise ChannelRefError(
                    f'[refs."{raw_ref}".metadata] must be a TOML table.'
                )
            self.bind(
                raw_ref,
                url=data.get("url"),
                store=data.get("store"),
                path=data.get("path"),
                metadata={
                    key: value
                    for key, value in data.items()
                    if key not in {"url", "store", "path"}
                },
            )

    def bind(
        self,
        ref: str | ChannelRef,
        *,
        url: str | None = None,
        store: str | None = None,
        path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        parsed = ChannelRef.parse(ref)
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ChannelRefError(f"Ref binding metadata must be a table: {parsed}")
        url = _normalize_text_target(parsed.value, "url", url)
        store = _normalize_text_target(parsed.value, "store", store)
        path = _normalize_text_target(parsed.value, "path", path)
        _validate_target(parsed.value, url=url, store=store, path=path)
        self._bindings[parsed.value] = ResolvedChannelRef(
            ref=parsed.value,
            resource_kind=parsed.resource_kind,
            name=parsed.name,
            url=url,
            store=store,
            path=path,
            metadata=copy_boundary_value(metadata or {}),
        )

    def resolve(self, ref: str | ChannelRef) -> ResolvedChannelRef:
        parsed = ChannelRef.parse(ref)
        try:
            binding = self._bindings[parsed.value]
        except KeyError as exc:
            raise ChannelRefError(f"Channel ref is not bound: {parsed}") from exc
        return _copy_binding(binding)

    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(self._bindings))

    def client(
        self,
        ref: str | ChannelRef,
        *,
        timeout: float | None = 30.0,
        transport: Any = None,
    ) -> RemoteStore:
        binding = self.resolve(ref)
        if binding.url is None:
            raise ChannelRefError(f"Channel ref is not bound to an HTTP URL: {binding.ref}")
        return RemoteStore(binding.url, timeout=timeout, transport=transport)

    def async_client(
        self,
        ref: str | ChannelRef,
        *,
        timeout: float | None = 30.0,
        transport: Any = None,
    ) -> AsyncRemoteStore:
        binding = self.resolve(ref)
        if binding.url is None:
            raise ChannelRefError(f"Channel ref is not bound to an HTTP URL: {binding.ref}")
        return AsyncRemoteStore(binding.url, timeout=timeout, transport=transport)

    def store_path(self, ref: str | ChannelRef) -> Path:
        binding = self.resolve(ref)
        target = binding.store if binding.store is not None else binding.path
        if target is None:
            raise ChannelRefError(f"Channel ref is not bound to a local store: {binding.ref}")
        if binding.store in self._stores:
            target = self._stores[binding.store].get("path")
            if target is None:
                raise ChannelRefError(
                    f"[stores.{binding.store}] has no path for channel ref: "
                    f"{binding.ref}"
                )
        return _resolve_under_root(self.root, target)

    def local_store(self, ref: str | ChannelRef) -> LocalStore:
        return LocalStore(self.store_path(ref))

    def stores(self) -> dict[str, dict[str, Any]]:
        return copy_boundary_value(self._stores)

    def store(self, name: str) -> dict[str, Any]:
        _validate_table_name(name, f"stores.{name}")
        try:
            return copy_boundary_value(self._stores[name])
        except KeyError as exc:
            raise KeyError(f"Store config is not bound: {name}") from exc


def _copy_binding(binding: ResolvedChannelRef) -> ResolvedChannelRef:
    return ResolvedChannelRef(
        ref=binding.ref,
        resource_kind=binding.resource_kind,
        name=binding.name,
        url=binding.url,
        store=binding.store,
        path=binding.path,
        metadata=copy_boundary_value(binding.metadata),
    )


def _load_toml(path: Path) -> dict[str, Any]:
    return _parse_toml(path.read_text(encoding="utf-8"))


def _parse_toml(text: str) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(text)


def _config_text_value(value: Any) -> str:
    if not isinstance(value, str):
        raise ChannelRefError("config text must be a string.")
    return value


def _config_target(path: str | Path) -> tuple[Path, Path]:
    target = Path(_path_value(path, "config path")).expanduser()
    if target.is_dir():
        root = target
        target = target / ".lllm" / "config.toml"
    else:
        root = target.parent.parent if target.parent.name == ".lllm" else target.parent
    return root.resolve(), target


def _is_channel_config_ref(ref: Any) -> bool:
    if not isinstance(ref, str) or not ref.strip():
        raise ChannelRefError("Ref binding key must be a non-empty string.")
    parsed = urlparse(ref)
    raw_parts = parsed.path.split("/")
    if (
        parsed.scheme != "lllm"
        or not parsed.netloc
        or parsed.params
        or parsed.query
        or parsed.fragment
        or len(raw_parts) != 4
        or raw_parts[0] != ""
    ):
        raise ChannelRefError(f"Ref binding key must be a lllm:// resource ref: {ref}")
    package, resource_kind, name = raw_parts[1:]
    for segment in (parsed.netloc, package, resource_kind, name):
        decoded = unquote(segment)
        if (
            decoded in {".", ".."}
            or decoded != segment
            or "%" in segment
            or not decoded.strip()
            or any(ch.isspace() for ch in decoded)
            or any(ch in decoded for ch in "/\\:")
        ):
            raise ChannelRefError(
                f"Ref binding key must use plain path segments: {ref}"
            )
    if resource_kind in _NON_CHANNEL_CONFIG_REF_SECTIONS:
        return False
    return True


def _table_of_tables(value: Any, name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ChannelRefError(f"[{name}] must be a TOML table.")
    result: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        if not isinstance(item, dict):
            raise ChannelRefError(f"[{name}.{key}] must be a TOML table.")
        if not isinstance(key, str):
            raise ChannelRefError(
                f"[{name}.{key}] must use a non-empty path-segment name."
            )
        _validate_table_name(key, f"{name}.{key}")
        item_copy = copy_boundary_value(item)
        _validate_store_table_values(key, item_copy)
        result[key] = item_copy
    return result


def _validate_table_name(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "%" in value
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\")
    ):
        raise ChannelRefError(f"[{label}] must use a non-empty path-segment name.")


def _validate_store_table_values(key: str, item: dict[str, Any]) -> None:
    if "metadata" in item and not isinstance(item["metadata"], dict):
        raise ChannelRefError(f"[stores.{key}.metadata] must be a TOML table.")
    if "path" not in item:
        return
    path = item["path"]
    if (
        not isinstance(path, str)
        or not path
        or path != path.strip()
        or any(ch.isspace() for ch in path)
    ):
        raise ChannelRefError(
            f"[stores.{key}] path must be a non-empty string without whitespace."
        )


def _normalize_text_target(ref: str, name: str, value: Any) -> str | None:
    if value is None:
        return None
    if (
        isinstance(value, str)
        and value
        and value == value.strip()
        and not any(ch.isspace() for ch in value)
    ):
        return value
    raise ChannelRefError(
        f"Ref binding target {name!r} must be a non-empty string without whitespace: {ref}"
    )


def _validate_target(
    ref: str,
    *,
    url: str | None,
    store: str | None,
    path: str | None,
) -> None:
    targets = {"url": url, "store": store, "path": path}
    active = [name for name, value in targets.items() if value is not None]
    if not active:
        raise ChannelRefError(f"Ref binding must declare one concrete target: {ref}")
    if len(active) > 1:
        targets_text = ", ".join(active)
        raise ChannelRefError(
            "Ref binding must declare only one concrete target, "
            f"got {targets_text}: {ref}"
        )
    if url is not None:
        _validate_url_target(ref, url)


def _validate_url_target(ref: str, value: str) -> None:
    parsed = urlparse(value)
    if (
        parsed.scheme in {"http", "https"}
        and parsed.netloc
        and not any(ch.isspace() for ch in value)
    ):
        return
    raise ChannelRefError(
        f"Ref binding target 'url' must be an absolute HTTP(S) URL: {ref}"
    )


def _resolve_under_root(root: Path, path: str) -> Path:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = root / target
    return target.resolve()


def _path_value(value: Any, label: str) -> str:
    try:
        text = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a non-empty path string") from exc
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    return text
