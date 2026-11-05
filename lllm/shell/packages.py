"""LLLM hub package integration for shell strategies."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import StrategyResource
from .strategy import Handler, Strategy


def strategy_from_package(
    path: str | Path,
    *,
    name: str | None = None,
    prefix: str | None = None,
    store: str | Path | None = None,
    bind: bool = True,
) -> Strategy:
    """Create a strategy from one LLLM hub package folder."""

    manifest = _load_manifest(path)
    strategy = Strategy(
        name or manifest.package.name,
        description=manifest.package.description,
        metadata={
            "package": manifest.identifier,
            "version": manifest.package.version,
            "source": str(manifest.base_dir or Path(path)),
            "card": _model_dump(manifest.card),
        },
    )
    return add_package(strategy, path, prefix=prefix, store=store, bind=bind)


def add_package(
    strategy: Strategy,
    path: str | Path,
    *,
    prefix: str | None = None,
    store: str | Path | None = None,
    bind: bool = True,
) -> Strategy:
    """Add package resources and local handlers to an existing strategy."""

    manifest = _load_manifest(path)
    base_dir = manifest.base_dir or _manifest_base(path)
    package_prefix = _resource_prefix(prefix)
    package_name = _local_name(package_prefix, manifest.package.name)
    strategy.add_resource(
        StrategyResource(
            kind="package",
            name=package_name,
            ref=f"lllm://{manifest.package.org}/{manifest.package.name}",
            description=manifest.package.description,
            metadata={
                "package": manifest.identifier,
                "version": manifest.package.version,
                "path": str(base_dir),
                "primary": manifest.package.primary,
                "card": _model_dump(manifest.card),
            },
        )
    )

    if manifest.config is not None:
        strategy.add_resource(
            _strategy_resource(
                manifest,
                "config",
                "default",
                description=manifest.config.description,
                metadata={
                    **manifest.config.metadata,
                    "schema": manifest.config.schema,
                    "defaults": manifest.config.defaults,
                },
                prefix=package_prefix,
            )
        )

    tactic_handlers: dict[str, Handler] = {}
    for name, resource in manifest.schemas.items():
        strategy.add_resource(
            _strategy_resource(
                manifest,
                "schema",
                name,
                entrypoint=resource.entry,
                description=resource.description,
                metadata=resource.metadata,
                prefix=package_prefix,
            )
        )

    for name, resource in manifest.tactics.items():
        handler = (
            _tactic_handler(resource.entry, base_dir=base_dir, name=name)
            if bind
            else None
        )
        if handler is not None:
            tactic_handlers[name] = handler
        strategy.add_resource(
            _strategy_resource(
                manifest,
                "tactic",
                name,
                entrypoint=resource.entry,
                description=resource.description,
                metadata={
                    **resource.metadata,
                    "runtime": resource.runtime,
                    "input": resource.input,
                    "output": resource.output,
                    "examples": list(resource.examples),
                },
                prefix=package_prefix,
            ),
            handler=handler,
        )

    channel_store = (
        _channel_store(manifest, store=store)
        if bind and manifest.channels
        else None
    )
    for name, resource in manifest.channels.items():
        handler = (
            _channel_handler(channel_store, name)
            if channel_store is not None
            else None
        )
        strategy.add_resource(
            _strategy_resource(
                manifest,
                "channel",
                name,
                description=resource.description,
                metadata={
                    **resource.metadata,
                    "schema": resource.schema,
                    "form": resource.form,
                    "store": str(getattr(channel_store, "root", "")) or None,
                },
                prefix=package_prefix,
            ),
            handler=handler,
        )

    for name, resource in manifest.snapshots.items():
        strategy.add_resource(
            _strategy_resource(
                manifest,
                "snapshot",
                name,
                description=resource.description,
                metadata={
                    **resource.metadata,
                    "schema": resource.schema,
                    "channel": resource.channel,
                },
                prefix=package_prefix,
            )
        )

    for name, resource in manifest.services.items():
        handler = None
        if bind and resource.tactic and resource.tactic in tactic_handlers:
            handler = tactic_handlers[resource.tactic]
        strategy.add_resource(
            _strategy_resource(
                manifest,
                "service",
                name,
                entrypoint=resource.entry,
                description=resource.description,
                metadata={
                    **resource.metadata,
                    "tactic": resource.tactic,
                    "tactics": list(resource.tactics),
                    "subscribes": list(resource.subscribes),
                    "publishes": list(resource.publishes),
                    "transport": resource.transport,
                },
                prefix=package_prefix,
            ),
            handler=handler,
        )

    for name, resource in manifest.runs.items():
        strategy.add_resource(
            _strategy_resource(
                manifest,
                "run",
                name,
                description=resource.description,
                metadata={
                    **resource.metadata,
                    "services": list(resource.services),
                    "tactics": list(resource.tactics),
                    "channels": list(resource.channels),
                    "snapshots": list(resource.snapshots),
                },
                prefix=package_prefix,
            )
        )

    for name, resource in manifest.docs.items():
        strategy.add_resource(
            _strategy_resource(
                manifest,
                "doc",
                name,
                description=resource.description,
                metadata={
                    **resource.metadata,
                    "path": resource.path,
                    "title": resource.title,
                },
                prefix=package_prefix,
            )
        )

    for name, resource in manifest.examples.items():
        strategy.add_resource(
            _strategy_resource(
                manifest,
                "example",
                name,
                description=resource.description,
                metadata={
                    **resource.metadata,
                    "path": resource.path,
                    "command": resource.command,
                },
                prefix=package_prefix,
            )
        )

    for name, resource in manifest.assets.items():
        strategy.add_resource(
            _strategy_resource(
                manifest,
                "asset",
                name,
                description=resource.description,
                metadata={
                    **resource.metadata,
                    "path": resource.path,
                    "media_type": resource.media_type,
                },
                prefix=package_prefix,
            )
        )

    return strategy


def _strategy_resource(
    manifest: Any,
    kind: str,
    name: str,
    *,
    entrypoint: str | None = None,
    description: str = "",
    metadata: Mapping[str, Any] | None = None,
    prefix: str | None = None,
) -> StrategyResource:
    return StrategyResource(
        kind=kind,
        name=_local_name(prefix, name),
        ref=manifest.ref(kind, name),
        entrypoint=entrypoint,
        description=description,
        metadata={
            "package": manifest.identifier,
            "version": manifest.package.version,
            "resource": name,
            **{
                key: value
                for key, value in dict(metadata or {}).items()
                if value is not None
            },
        },
    )


def _tactic_handler(entrypoint: str, *, base_dir: Path, name: str) -> Handler:
    async def handler(
        input_value: Any = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        tactic = _load_tactic(entrypoint, base_dir=base_dir, name=name)
        if hasattr(tactic, "arun") and callable(tactic.arun):
            return await tactic.arun(input_value, context=_call_context(context))
        if hasattr(tactic, "run") and callable(tactic.run):
            return tactic.run(input_value, context=_call_context(context))
        if callable(tactic):
            result = _call_plain_tactic(tactic, input_value, context=context)
            if inspect.isawaitable(result):
                return await result
            return result
        raise TypeError(f"Tactic entrypoint is not callable: {entrypoint}")

    return handler


def _load_tactic(entrypoint: str, *, base_dir: Path, name: str) -> Any:
    from ..hub import import_entrypoint
    from ..protocol import Tactic
    from ..runtimes import as_tactic

    value = import_entrypoint(entrypoint, base_dir=base_dir)
    if isinstance(value, type) and issubclass(value, Tactic):
        return value(name=name)
    if isinstance(value, Tactic):
        return value
    if callable(value) and _can_call_without_args(value):
        produced = value()
        if isinstance(produced, type) and issubclass(produced, Tactic):
            return produced(name=name)
        if isinstance(produced, Tactic):
            return produced
        if hasattr(produced, "run") and callable(produced.run):
            return as_tactic(produced.run, name=name)
        if callable(produced):
            return as_tactic(produced, name=name)
    if hasattr(value, "run") and callable(value.run):
        return as_tactic(value.run, name=name)
    if callable(value):
        return as_tactic(value, name=name)
    return value


def _channel_store(manifest: Any, *, store: str | Path | None) -> Any:
    from ..network import ChannelExistsError, LocalStore

    base_dir = manifest.base_dir or Path.cwd()
    store_root = (
        Path(store).expanduser() if store is not None else base_dir / ".lllm" / "network"
    )
    local_store = LocalStore(store_root)
    for name, channel in manifest.channels.items():
        try:
            local_store.create_channel(
                {
                    "name": name,
                    "schema": channel.schema,
                    "form": channel.form,
                    "description": channel.description,
                    "metadata": channel.metadata,
                }
            )
        except ChannelExistsError:
            pass
    return local_store


def _channel_handler(store: Any, name: str) -> Handler:
    async def handler(
        input_value: Any = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        request = input_value if isinstance(input_value, Mapping) else {}
        op = str(request.get("op") or "query")
        if op == "info":
            return store.get_channel(name)
        if op == "append":
            event = {
                "channel": name,
                "payload": request.get("payload"),
                "kind": request.get("kind") or "event",
                "source": request.get("source") or "lllm.shell",
                "schema": request.get("schema"),
                "metadata": request.get("metadata") or {},
                "correlation_id": request.get("correlation_id"),
                "parent_ids": tuple(request.get("parent_ids") or ()),
            }
            return store.append_event(event)
        if op == "query":
            return list(
                store.query_events(
                    name,
                    after_cursor=_non_negative_int(request.get("after_cursor", 0)),
                    limit=_positive_int(request.get("limit", 100)),
                    kind=request.get("kind"),
                )
            )
        raise ValueError("channel op must be one of: append, info, query")

    return handler


def _load_manifest(path: str | Path) -> Any:
    from ..hub import load_manifest

    return load_manifest(path)


def _manifest_base(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if target.is_file():
        return target.parent
    return target


def _resource_prefix(prefix: str | None) -> str | None:
    if prefix is None or prefix == "":
        return None
    if (
        not isinstance(prefix, str)
        or any(ch.isspace() for ch in prefix)
        or any(ch in prefix for ch in "/:\\")
    ):
        raise ValueError("package prefix must be a path-segment friendly string")
    return prefix


def _local_name(prefix: str | None, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


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


def _call_plain_tactic(
    tactic: Any,
    input_value: Any,
    *,
    context: Mapping[str, Any] | None,
) -> Any:
    signature = _signature(tactic)
    kwargs: dict[str, Any] = {}
    if signature is not None and _accepts_context(signature):
        kwargs["context"] = _call_context(context)
    if signature is not None and not _can_accept_input(signature):
        return tactic(**kwargs)
    return tactic(input_value, **kwargs)


def _signature(value: Any) -> inspect.Signature | None:
    try:
        return inspect.signature(value)
    except (TypeError, ValueError):
        return None


def _accepts_context(signature: inspect.Signature) -> bool:
    if "context" in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _can_accept_input(signature: inspect.Signature) -> bool:
    for parameter in signature.parameters.values():
        if parameter.name == "context":
            continue
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            return True
    return any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )


def _call_context(context: Mapping[str, Any] | None) -> Any:
    if context is None:
        return None
    try:
        from ..protocol import CallContext
    except ImportError:
        return dict(context)
    if isinstance(context, CallContext):
        return context
    return CallContext(metadata=dict(context))


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("after_cursor must be a non-negative integer")
    return value


def _positive_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("limit must be a positive integer")
    return value


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump()
    return value
