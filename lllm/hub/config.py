"""Local `.lllm/config.toml` ref resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..protocol._validation import copy_boundary_value, metadata_mapping_value
from ..protocol._validation import is_sensitive_metadata_key as _is_sensitive_metadata_key
from .manifest import require_path_value
from .refs import validate_resource_ref


@dataclass(frozen=True)
class ResolvedRef:
    ref: str
    url: str | None = None
    store: str | None = None
    path: str | None = None
    object: Any = None
    metadata: dict[str, Any] | None = None


class LocalConfigResolver:
    """Resolve `lllm://...` refs from local config and object bindings."""

    def __init__(self) -> None:
        self._bindings: dict[str, ResolvedRef] = {}
        self._settings: dict[str, Any] = {}
        self._services: dict[str, dict[str, Any]] = {}
        self._stores: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> "LocalConfigResolver":
        resolver = cls()
        target = Path(require_path_value(path, "config path"))
        if target.is_dir():
            target = target / ".lllm" / "config.toml"
        data = _load_toml(target)
        refs = data.get("refs", {})
        if not isinstance(refs, dict):
            raise ValueError("[refs] must be a TOML table.")
        settings = data.get("settings", {})
        if not isinstance(settings, dict):
            raise ValueError("[settings] must be a TOML table.")
        resolver._settings = copy_boundary_value(settings)
        resolver._services = _table_of_tables(data.get("services", {}), "services")
        resolver._stores = _table_of_tables(data.get("stores", {}), "stores")
        for ref, binding in refs.items():
            if not isinstance(binding, dict):
                raise ValueError(f"Ref binding must be a table: {ref}")
            _validate_serialized_ref_binding(ref, binding)
            resolver.bind(
                ref,
                url=binding.get("url"),
                store=binding.get("store"),
                path=binding.get("path"),
                metadata=_ref_binding_metadata(ref, binding),
            )
        return resolver

    @classmethod
    def from_text(cls, text: str, *, root: str | Path | None = None) -> "LocalConfigResolver":
        text_value = _config_text_value(text)
        root_value = "." if root is None else root
        root_path = Path(require_path_value(root_value, "config root"))
        target = root_path / ".lllm" / "config.toml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text_value, encoding="utf-8")
        return cls.from_file(root_path)

    def bind(
        self,
        ref: str,
        *,
        url: str | None = None,
        store: str | None = None,
        path: str | None = None,
        object: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        validate_resource_ref(ref)
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError(f"Ref binding metadata must be a table: {ref}")
        try:
            metadata_value = (
                {} if metadata is None else metadata_mapping_value("metadata", metadata)
            )
        except TypeError as exc:
            raise ValueError(f"{exc}: {ref}") from exc
        _reject_sensitive_metadata(metadata_value, f'refs."{ref}".metadata')
        url = _normalize_text_target(ref, "url", url)
        store = _normalize_text_target(ref, "store", store)
        path = _normalize_text_target(ref, "path", path)
        _validate_target(ref, url=url, store=store, path=path, object=object)
        self._bindings[ref] = ResolvedRef(
            ref=ref,
            url=url,
            store=store,
            path=path,
            object=object,
            metadata=metadata_value,
        )

    def resolve(self, ref: str) -> ResolvedRef:
        validate_resource_ref(ref)
        try:
            binding = self._bindings[ref]
        except KeyError as exc:
            raise KeyError(f"Ref is not bound: {ref}") from exc
        return _copy_binding(binding)

    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(self._bindings))

    def settings(self) -> dict[str, Any]:
        return copy_boundary_value(self._settings)

    def setting(self, key: str, default: Any = None) -> Any:
        return copy_boundary_value(self._settings.get(key, default))

    def services(self) -> dict[str, dict[str, Any]]:
        return copy_boundary_value(self._services)

    def service(self, name: str) -> dict[str, Any]:
        _validate_table_name(name, f"services.{name}")
        try:
            return copy_boundary_value(self._services[name])
        except KeyError as exc:
            raise KeyError(f"Service config is not bound: {name}") from exc

    def stores(self) -> dict[str, dict[str, Any]]:
        return copy_boundary_value(self._stores)

    def store(self, name: str) -> dict[str, Any]:
        _validate_table_name(name, f"stores.{name}")
        try:
            return copy_boundary_value(self._stores[name])
        except KeyError as exc:
            raise KeyError(f"Store config is not bound: {name}") from exc


def _copy_binding(binding: ResolvedRef) -> ResolvedRef:
    return ResolvedRef(
        ref=binding.ref,
        url=binding.url,
        store=binding.store,
        path=binding.path,
        object=binding.object,
        metadata=(
            copy_boundary_value(binding.metadata) if binding.metadata is not None else None
        ),
    )


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib  # type: ignore[no-redef]
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _config_text_value(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("config text must be a string.")
    return value


def _ref_binding_metadata(ref: str, binding: dict[str, Any]) -> dict[str, Any]:
    if "metadata" in binding:
        metadata = binding["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError(f'[refs."{ref}".metadata] must be a TOML table.')
    else:
        metadata = {}
    extras = {
        key: value
        for key, value in binding.items()
        if key not in {"url", "store", "path", "object", "metadata"}
    }
    return {**extras, **metadata}


def _validate_serialized_ref_binding(ref: str, binding: dict[str, Any]) -> None:
    if "object" in binding:
        raise ValueError(
            "Ref binding target 'object' is only supported for in-process "
            f"LocalConfigResolver.bind(..., object=...) bindings: {ref}"
        )


def _table_of_tables(value: Any, name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table.")
    result: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        if not isinstance(item, dict):
            raise ValueError(f"[{name}.{key}] must be a TOML table.")
        if not isinstance(key, str):
            raise ValueError(f"[{name}.{key}] must use a non-empty path-segment name.")
        key_text = key
        _validate_table_name(key_text, f"{name}.{key_text}")
        item_copy = copy_boundary_value(item)
        _validate_table_values(name, key_text, item_copy)
        result[key_text] = item_copy
    return result


def _validate_table_name(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\;%")
    ):
        raise ValueError(f"[{label}] must use a non-empty path-segment name.")


def _validate_table_values(section: str, key: str, item: dict[str, Any]) -> None:
    if "metadata" in item and not isinstance(item["metadata"], dict):
        raise ValueError(f"[{section}.{key}.metadata] must be a TOML table.")
    _reject_sensitive_metadata(item, f"{section}.{key}")
    if section == "services" and "port" in item:
        port = item["port"]
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not (1 <= port <= 65535)
        ):
            raise ValueError(
                f"[services.{key}] port must be an integer between 1 and 65535."
            )
    if section == "stores" and "path" in item:
        path = item["path"]
        if (
            not isinstance(path, str)
            or not path
            or path != path.strip()
            or any(ch.isspace() for ch in path)
        ):
            raise ValueError(
                f"[stores.{key}] path must be a non-empty string without whitespace."
            )
        _validate_local_path_target(f"stores.{key}", "path", path)


def _reject_sensitive_metadata(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _is_sensitive_metadata_key(key):
                raise ValueError(
                    f"[{label}] must not include raw secret key {key!r}."
                )
            _reject_sensitive_metadata(item, label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_sensitive_metadata(item, label)


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
    raise ValueError(
        f"Ref binding target {name!r} must be a non-empty string without whitespace: {ref}"
    )


def _validate_target(
    ref: str,
    *,
    url: str | None,
    store: str | None,
    path: str | None,
    object: Any,
) -> None:
    targets = {
        "url": url,
        "store": store,
        "path": path,
        "object": object,
    }
    active = [name for name, value in targets.items() if value is not None]
    if not active:
        raise ValueError(f"Ref binding must declare one concrete target: {ref}")
    if len(active) == 1:
        _validate_text_target(ref, active[0], targets[active[0]])
        return
    targets_text = ", ".join(active)
    raise ValueError(
        "Ref binding must declare only one concrete target, "
        f"got {targets_text}: {ref}"
    )


def _validate_text_target(ref: str, name: str, value: Any) -> None:
    if name == "object":
        return
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Ref binding target {name!r} must be a non-empty string: {ref}"
        )
    if name == "url":
        _validate_url_target(ref, value)
        return
    if name in {"store", "path"}:
        _validate_local_path_target(ref, name, value)
        return


def _validate_local_path_target(label: str, name: str, value: str) -> None:
    bad = value.startswith(("/", "~")) or any(ch in value for ch in "\\:%?#;")
    parts = value.split("/")
    if value == ".":
        return
    if bad or any(part in {"", ".", ".."} or part.startswith("~") for part in parts):
        if label.startswith("lllm://"):
            prefix = f"Ref binding target {name!r}"
            suffix = f": {label}"
        else:
            prefix = f"[{label}] {name}"
            suffix = ""
        raise ValueError(
            f"{prefix} must be a portable relative path without traversal "
            f"or URL syntax{suffix}"
        )


def _validate_url_target(ref: str, value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"Ref binding target 'url' must be an absolute HTTP(S) URL: {ref}"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            f"Ref binding target 'url' must not include embedded credentials: {ref}"
        )
    if (
        ";" in parsed.netloc
        or parsed.params
        or ";" in parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Ref binding target 'url' must not include URL params, query "
            f"or fragment parts: {ref}"
        )
