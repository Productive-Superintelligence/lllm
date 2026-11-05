"""Load LLLM packages or Python entrypoints as FastAPI services."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .credentials import ApiKeyRequirement, collect_manifest_api_key_requirements
from .hub import import_entrypoint, load_manifest
from .protocol import Tactic
from .runtimes import coerce_tactic
from .services import create_tactic_app


class LaunchError(ValueError):
    """Raised when a target cannot be launched as a service."""


@dataclass(frozen=True)
class LaunchApp:
    """Resolved launch target."""

    app: Any
    label: str
    kind: str
    port: int = 8000


def load_launch_app(
    target: str | Path,
    *,
    resource: str | None = None,
    store: str | Path | None = None,
) -> LaunchApp:
    """Resolve a package path, manifest path, or entrypoint to an ASGI app."""

    target_text = _target_text(target)
    path = Path(target_text).expanduser()
    if path.exists() or path.name == "lllm.toml":
        return _load_package_app(path, resource=resource, store=store)
    if ":" in target_text:
        value = import_entrypoint(target_text)
        app = _coerce_entrypoint_app(value, label=target_text)
        return LaunchApp(app=app, label=target_text, kind="entrypoint")
    raise LaunchError(
        "launch target must be a package path, lllm.toml path, or module:attribute entrypoint"
    )


def load_launch_api_key_requirements(
    target: str | Path,
    *,
    resource: str | None = None,
) -> tuple[ApiKeyRequirement, ...]:
    """Resolve API-key requirements declared by a launchable package."""

    target_text = _target_text(target)
    path = Path(target_text).expanduser()
    if path.exists() or path.name == "lllm.toml":
        manifest = load_manifest(path)
        section, name = _package_resource(manifest, resource)
        return collect_manifest_api_key_requirements(
            manifest,
            section=section,
            name=name,
        )
    return ()


def _load_package_app(
    path: Path,
    *,
    resource: str | None,
    store: str | Path | None,
) -> LaunchApp:
    manifest = load_manifest(path)
    section, name = _package_resource(manifest, resource)
    port = _resource_port(manifest, section, name)
    label = f"{manifest.identifier}:{section}.{name}"
    base_dir = manifest.base_dir
    if base_dir is None:
        raise LaunchError("package manifest has no base directory")

    if section == "services":
        service = manifest.services.get(name)
        if service is None:
            raise LaunchError(f"service not found in package: {name}")
        app = _service_app(manifest, name, service, base_dir)
        return LaunchApp(app=app, label=label, kind="service", port=port)

    if section == "tactics":
        tactic = manifest.tactics.get(name)
        if tactic is None:
            raise LaunchError(f"tactic not found in package: {name}")
        app = _tactic_app(tactic.entry, base_dir=base_dir, name=name)
        return LaunchApp(app=app, label=label, kind="tactic", port=port)

    if section == "channels":
        app = _channel_app(manifest, store=store)
        return LaunchApp(app=app, label=label, kind="channel", port=port)

    if section == "runs":
        run = manifest.runs.get(name)
        if run is None:
            raise LaunchError(f"run not found in package: {name}")
        if len(run.services) == 1:
            return _load_package_app(
                path,
                resource=f"services.{run.services[0]}",
                store=store,
            )
        raise LaunchError(
            "run resources with multiple services need an explicit --resource services.NAME"
        )

    raise LaunchError(
        f"resource {section}.{name} is package metadata, not a launchable service"
    )


def _package_resource(manifest: Any, resource: str | None) -> tuple[str, str]:
    selected = resource or manifest.package.primary
    if not selected:
        selected = _first_resource(manifest)
    section, sep, name = str(selected).partition(".")
    if not sep or not section or not name:
        raise LaunchError("resource must have shape section.name")
    return section, name


def _first_resource(manifest: Any) -> str:
    for section, resources in (
        ("services", manifest.services),
        ("tactics", manifest.tactics),
        ("channels", manifest.channels),
        ("runs", manifest.runs),
    ):
        if resources:
            return f"{section}.{next(iter(resources))}"
    raise LaunchError("package does not declare a launchable resource")


def _service_app(manifest: Any, name: str, service: Any, base_dir: Path) -> Any:
    errors: list[str] = []
    if service.entry:
        try:
            value = import_entrypoint(service.entry, base_dir=base_dir)
            return _coerce_service_app(
                value,
                label=f"services.{name}",
                allow_payload=not bool(service.tactic),
            )
        except Exception as exc:
            errors.append(str(exc))
    if service.tactic:
        tactic = manifest.tactics.get(service.tactic)
        if tactic is None:
            raise LaunchError(f"service {name} references missing tactic {service.tactic}")
        return _tactic_app(tactic.entry, base_dir=base_dir, name=service.tactic)
    detail = f": {'; '.join(errors)}" if errors else ""
    raise LaunchError(f"service {name} did not resolve to a FastAPI app{detail}")


def _tactic_app(entry: str, *, base_dir: Path, name: str) -> Any:
    value = import_entrypoint(entry, base_dir=base_dir)
    tactic = _coerce_tactic(value, name=name)
    return create_tactic_app(tactic)


def _channel_app(manifest: Any, *, store: str | Path | None) -> Any:
    from .network.resources import open_package_store
    from .network.server import create_app

    return create_app(open_package_store(manifest, store=store))


def _coerce_entrypoint_app(value: Any, *, label: str) -> Any:
    if _is_asgi_app(value):
        return value
    if callable(value) and _can_call_without_args(value):
        produced = value()
        if _is_asgi_app(produced):
            return produced
        if _looks_like_tactic(produced):
            return _tactic_to_app(produced, name=label.rsplit(":", 1)[-1])
        return _constant_app(produced, label=label)
    if _looks_like_tactic(value) or callable(value):
        return _tactic_to_app(value, name=label.rsplit(":", 1)[-1])
    raise LaunchError(f"entrypoint did not resolve to an ASGI app or tactic: {label}")


def _coerce_service_app(value: Any, *, label: str, allow_payload: bool) -> Any:
    if _is_asgi_app(value):
        return value
    if callable(value) and _can_call_without_args(value):
        produced = value()
        if _is_asgi_app(produced):
            return produced
        if _looks_like_tactic(produced):
            return _tactic_to_app(produced, name=label.rsplit(".", 1)[-1])
        if allow_payload:
            return _constant_app(produced, label=label)
    raise LaunchError(f"{label} did not return an ASGI app or service payload")


def _coerce_tactic(value: Any, *, name: str) -> Any:
    try:
        return coerce_tactic(value, name=name)
    except TypeError as exc:
        raise LaunchError(f"entrypoint did not resolve to a tactic: {name}") from exc


def _tactic_to_app(value: Any, *, name: str) -> Any:
    return create_tactic_app(_coerce_tactic(value, name=name))


def _constant_app(value: Any, *, label: str) -> Any:
    try:
        from fastapi import FastAPI
        from fastapi.encoders import jsonable_encoder
    except ImportError as exc:  # pragma: no cover
        raise LaunchError("install fastapi to launch service payloads") from exc

    app = FastAPI(title=label, version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "service": label}

    @app.post("/run")
    async def run(request: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "output": jsonable_encoder(value),
            "service": label,
            "input": jsonable_encoder((request or {}).get("input")),
        }

    return app


def _looks_like_tactic(value: Any) -> bool:
    return (
        isinstance(value, Tactic)
        or (isinstance(value, type) and issubclass(value, Tactic))
        or _has_run_method(value)
    )


def _has_run_method(value: Any) -> bool:
    return value is not None and callable(getattr(value, "run", None))


def _is_asgi_app(value: Any) -> bool:
    return callable(value) and hasattr(value, "routes") and hasattr(value, "router")


def _can_call_without_args(value: Any) -> bool:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.default is inspect.Parameter.empty and parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            return False
    return True


def _resource_port(manifest: Any, section: str, name: str) -> int:
    resource = getattr(manifest, section, {}).get(name)
    metadata = getattr(resource, "metadata", {}) if resource is not None else {}
    port = metadata.get("port") if isinstance(metadata, dict) else None
    if isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535:
        return port
    if section == "services":
        return 8000
    if section == "tactics":
        for service in manifest.services.values():
            if service.tactic == name:
                service_port = _metadata_port(service.metadata)
                if service_port is not None:
                    return service_port
    return 8000


def _metadata_port(metadata: Any) -> int | None:
    if not isinstance(metadata, dict):
        return None
    port = metadata.get("port")
    if isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535:
        return port
    return None


def _target_text(target: str | Path) -> str:
    try:
        text = str(target)
    except TypeError as exc:
        raise LaunchError("launch target must be a path or entrypoint string") from exc
    if not text or text != text.strip():
        raise LaunchError("launch target must be a non-empty path or entrypoint string")
    return text
