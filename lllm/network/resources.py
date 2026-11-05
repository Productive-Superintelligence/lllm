"""Export channel and snapshot metadata in the shape the hub indexes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .core import Channel, ChannelExistsError, Snapshot
from ..protocol._validation import copy_boundary_value
from .server.endpoints import StoreEndpointSpec, endpoint_specs


def channel_resource(
    channel: Channel,
    *,
    custom_endpoints: Sequence[Callable[..., Any]] = (),
) -> dict[str, Any]:
    """Return manifest/card-friendly metadata for a channel."""

    return {
        "name": channel.name,
        "schema": channel.schema,
        "form": channel.form,
        "description": channel.description,
        "metadata": copy_boundary_value(channel.metadata),
        "endpoints": [
            _endpoint_metadata(spec) for _, spec in endpoint_specs(custom_endpoints)
        ],
    }


def snapshot_resource(
    snapshot: Snapshot,
    *,
    description: str = "",
    custom_endpoints: Sequence[Callable[..., Any]] = (),
) -> dict[str, Any]:
    """Return manifest/card-friendly metadata for a snapshot."""

    return {
        "name": snapshot.name,
        "schema": snapshot.schema,
        "channel": snapshot.channel,
        "description": description,
        "metadata": copy_boundary_value(snapshot.metadata),
        "endpoints": [
            _endpoint_metadata(spec) for _, spec in endpoint_specs(custom_endpoints)
        ],
    }


def _endpoint_metadata(spec: StoreEndpointSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "method": spec.method,
        "path": spec.path,
        "scope": spec.scope,
        "description": spec.description,
        "tags": list(spec.tags),
    }


def open_package_store(manifest: Any, *, store: str | Path | None = None) -> Any:
    """Open the local store for a package and create its declared channels.

    ``store`` defaults to ``.lllm/network`` inside the package folder. Channels
    declared in ``[channels.<name>]`` that already exist are left as they are.
    """

    from .stores import LocalStore

    base_dir = manifest.base_dir or Path.cwd()
    root = Path(store).expanduser() if store is not None else base_dir / ".lllm" / "network"
    local_store = LocalStore(root)
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
