"""LLLM hub package integration for shell strategies."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..protocol.refs import tactic_ref_value
from .models import StrategyResource
from .strategy import Strategy, StrategyStore


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
    """Add package resources and local handlers to an existing strategy.

    Resource names are ``<prefix>.<name>`` with a ``prefix``. A package whose
    names clash with resources already in the strategy is rejected before
    anything is added; mount it with a prefix to keep the names apart.

    With ``bind`` (the default), tactics get handlers that import the package
    on first use, and channels are backed by the strategy's one store under
    their resource names, so prefixed packages never share a channel. If the
    strategy has no store yet, the package opens one at ``store`` (default
    ``<package>/.lllm/network``); packages mounted later share it, and a
    ``store`` other than the strategy's raises ``ValueError``. The store is
    opened, and the channels created, on first use, never while building or
    inspecting. ``bind=False`` adds metadata only.
    """

    manifest = _load_manifest(path)
    base_dir = manifest.base_dir or _manifest_base(path)
    package_prefix = _resource_prefix(prefix)
    package_name = _local_name(package_prefix, manifest.package.name)
    _check_names(strategy, manifest, package_prefix)
    if bind and manifest.channels and (strategy.store is None or store is not None):
        root = base_dir / ".lllm" / "network" if store is None else store
        strategy.attach_store(StrategyStore(root=root))
    store_root = strategy.store.root if bind and strategy.store is not None else None
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

    tactic_handlers: dict[str, PackageTactic] = {}
    for name, resource in manifest.tactics.items():
        handler = (
            PackageTactic(
                resource.entry,
                base_dir=base_dir,
                name=name,
                ref=manifest.ref("tactic", name),
            )
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

    for name, resource in manifest.channels.items():
        channel = {
            "schema": resource.schema,
            "form": resource.form,
            "description": resource.description,
            "metadata": resource.metadata,
        }
        strategy._add_resource(
            _strategy_resource(
                manifest,
                "channel",
                name,
                description=resource.description,
                metadata={
                    **resource.metadata,
                    "schema": resource.schema,
                    "form": resource.form,
                    "store": None if store_root is None else str(store_root),
                },
                prefix=package_prefix,
            ),
            channel=channel if bind else None,
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


class PackageTactic:
    """A package tactic entrypoint, imported on first use.

    Building or inspecting a strategy never imports package code; serving it
    (or invoking the resource) does. A loaded tactic without a ``package_ref``
    gets ``ref``, the manifest ref ``lllm://org/package/tactics/name``, on a
    copy: the imported object may be shared with other packages or modules,
    and it is left as it is. A ``ref`` that is not a valid tactic ref (the hub
    accepts tactic names, like ``a#b``, that a ref cannot carry) is not used.
    """

    def __init__(
        self,
        entrypoint: str,
        *,
        base_dir: Path,
        name: str,
        ref: str | None = None,
    ) -> None:
        self.entrypoint = entrypoint
        self.base_dir = base_dir
        self.name = name
        self.ref = _tactic_ref(ref)
        self._tactic: Any = None

    def load(self) -> Any:
        if self._tactic is None:
            tactic = _load_tactic(self.entrypoint, base_dir=self.base_dir, name=self.name)
            if self.ref is not None and tactic.package_ref is None:
                tactic = copy.copy(tactic)
                tactic.package_ref = self.ref
            self._tactic = tactic
        return self._tactic

    async def __call__(
        self,
        input_value: Any = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        return await self.load().arun(input_value, context=_call_context(context))


def _load_tactic(entrypoint: str, *, base_dir: Path, name: str) -> Any:
    from ..hub import import_entrypoint
    from ..runtimes import coerce_tactic

    return coerce_tactic(import_entrypoint(entrypoint, base_dir=base_dir), name=name)


def _tactic_ref(ref: str | None) -> str | None:
    if ref is None:
        return None
    try:
        return tactic_ref_value(ref)
    except ValueError:
        return None


def _check_names(strategy: Strategy, manifest: Any, prefix: str | None) -> None:
    sections = (
        manifest.schemas,
        manifest.tactics,
        manifest.channels,
        manifest.snapshots,
        manifest.services,
        manifest.runs,
        manifest.docs,
        manifest.examples,
        manifest.assets,
    )
    names = [manifest.package.name, *(["default"] if manifest.config is not None else [])]
    names += [name for section in sections for name in section]
    existing = {resource.name for resource in strategy.resources}
    taken = sorted({_local_name(prefix, name) for name in names} & existing)
    if taken:
        raise ValueError(
            f"Cannot mount package {manifest.identifier}: strategy {strategy.name!r} "
            f"already has resources named {', '.join(taken)}. Mount it with a "
            "prefix to keep the names apart."
        )


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


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump()
    return value
