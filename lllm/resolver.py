"""Resolve any ``lllm://`` ref from one local config.

A project keeps its bindings in ``.lllm/config.toml``::

    [refs."lllm://demo/briefs/tactics/brief"]
    url = "http://127.0.0.1:8000"

    [refs."lllm://demo/briefs/channels/events"]
    store = ".lllm/network"

    [settings]
    region = "us"

:class:`Resolver` reads that file once and hands back live objects: tactics
(local or :class:`~lllm.services.RemoteTactic`) for ``/tactics/`` refs and
channel stores (:class:`~lllm.network.LocalStore` or
:class:`~lllm.network.RemoteStore`) for ``/channels/`` and ``/snapshots/``
refs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .protocol import CallContext, Tactic
from .protocol.refs import parse_resource_ref
from .services import TacticResolver

if TYPE_CHECKING:
    from .hub.config import LocalConfigResolver
    from .network import ChannelResolver, LocalStore, RemoteStore


class Resolver:
    """One entry point for tactic, channel, and snapshot refs plus settings."""

    def __init__(
        self,
        *,
        tactics: TacticResolver | None = None,
        channels: "ChannelResolver | None" = None,
        config: "LocalConfigResolver | None" = None,
    ) -> None:
        from .hub.config import LocalConfigResolver
        from .network import ChannelResolver

        self.tactics = tactics if tactics is not None else TacticResolver()
        self.channels = channels if channels is not None else ChannelResolver()
        self.config = config if config is not None else LocalConfigResolver()

    @classmethod
    def from_config(cls, path: str | Path = ".") -> "Resolver":
        """Load ``.lllm/config.toml`` from a project folder or a file path."""

        from .hub.config import LocalConfigResolver
        from .network import ChannelResolver

        return cls(
            tactics=TacticResolver.from_config(path),
            channels=ChannelResolver.from_config(path),
            config=LocalConfigResolver.from_file(path),
        )

    def register(self, ref: str, tactic: Tactic[Any, Any]) -> None:
        """Bind a tactic ref to an in-process tactic."""

        self.tactics.register(ref, tactic)

    def tactic(self, ref: str) -> Tactic[Any, Any]:
        return self.tactics.resolve(ref)

    def store(self, ref: str) -> "LocalStore | RemoteStore":
        """Return the store behind a channel or snapshot ref."""

        binding = self.channels.resolve(ref)
        if binding.url is not None:
            return self.channels.client(ref)
        return self.channels.local_store(ref)

    def resolve(self, ref: str) -> Any:
        """Resolve a ref by its section: tactics, channels, or snapshots."""

        section = parse_resource_ref(ref).resource_kind
        if section == "tactics":
            return self.tactic(ref)
        if section in {"channels", "snapshots"}:
            return self.store(ref)
        raise ValueError(
            f"Resolver can resolve tactics, channels, and snapshots, not /{section}/: {ref}"
        )

    def run(
        self,
        ref: str,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.tactics.run(ref, input_value, context=context, **kwargs)

    async def arun(
        self,
        ref: str,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self.tactics.arun(ref, input_value, context=context, **kwargs)

    def setting(self, key: str, default: Any = None) -> Any:
        return self.config.setting(key, default)

    def refs(self) -> tuple[str, ...]:
        return tuple(sorted({*self.tactics.refs(), *self.channels.refs()}))
