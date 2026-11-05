"""Strategy abstraction for application packages."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from .models import ResourceKind, StrategyInfo, StrategyResource

Handler = Callable[..., Any | Awaitable[Any]]


class Strategy:
    """Encapsulate package resources and expose one application-level boundary."""

    def __init__(
        self,
        name: str,
        *,
        description: str = "",
        resources: list[StrategyResource] | tuple[StrategyResource, ...] = (),
        runner: Handler | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self._resources: dict[str, StrategyResource] = {}
        self._handlers: dict[str, Handler] = {}
        self._runner = runner
        self.metadata = dict(metadata or {})
        for resource in resources:
            self.add_resource(resource)

    @property
    def resources(self) -> tuple[StrategyResource, ...]:
        return tuple(self._resources.values())

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name=self.name,
            description=self.description,
            resources=self.resources,
            metadata=dict(self.metadata),
        )

    def runner(self, fn: Handler) -> Handler:
        """Decorator or setter for the strategy's main run handler."""

        self._runner = fn
        return fn

    def add_resource(
        self,
        resource: StrategyResource,
        *,
        handler: Handler | None = None,
    ) -> "Strategy":
        if resource.name in self._resources:
            raise ValueError(f"Strategy resource already exists: {resource.name}")
        self._resources[resource.name] = resource
        if handler is not None:
            self._handlers[resource.name] = handler
        return self

    def resource(
        self,
        kind: ResourceKind,
        name: str,
        *,
        ref: str | None = None,
        entrypoint: str | None = None,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
        handler: Handler | None = None,
    ) -> "Strategy":
        return self.add_resource(
            StrategyResource(
                kind=kind,
                name=name,
                ref=ref,
                entrypoint=entrypoint,
                description=description,
                metadata=dict(metadata or {}),
            ),
            handler=handler,
        )

    def tactic(self, name: str, **kwargs: Any) -> "Strategy":
        return self.resource("tactic", name, **kwargs)

    def channel(self, name: str, **kwargs: Any) -> "Strategy":
        return self.resource("channel", name, **kwargs)

    def service(self, name: str, **kwargs: Any) -> "Strategy":
        return self.resource("service", name, **kwargs)

    def package(self, name: str, **kwargs: Any) -> "Strategy":
        return self.resource("package", name, **kwargs)

    def snapshot(self, name: str, **kwargs: Any) -> "Strategy":
        return self.resource("snapshot", name, **kwargs)

    def use_package(
        self,
        path: str | Path,
        *,
        prefix: str | None = None,
        store: str | Path | None = None,
        bind: bool = True,
    ) -> "Strategy":
        """Import resources from a LLLM hub package manifest."""

        from .packages import add_package

        return add_package(self, path, prefix=prefix, store=store, bind=bind)

    async def arun(
        self,
        input_value: Any = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        if self._runner is None:
            return {
                "strategy": self.name,
                "input": input_value,
                "resources": [resource.model_dump() for resource in self.resources],
            }
        return await _call(self._runner, input_value, context=context)

    async def invoke_resource(
        self,
        name: str,
        input_value: Any = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        if name not in self._resources:
            raise KeyError(f"Strategy resource not found: {name}")
        handler = self._handlers.get(name)
        if handler is None:
            resource = self._resources[name]
            return {
                "resource": resource.model_dump(),
                "input": input_value,
                "context": dict(context or {}),
            }
        return await _call(handler, input_value, context=context)


async def _call(
    handler: Handler,
    input_value: Any,
    *,
    context: Mapping[str, Any] | None,
) -> Any:
    kwargs: dict[str, Any] = {}
    signature = _signature(handler)
    if signature is not None and _accepts_context(signature):
        kwargs["context"] = dict(context or {})
    if signature is not None and _can_accept_input(signature):
        result = handler(input_value, **kwargs)
    else:
        result = handler(**kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _signature(handler: Handler) -> inspect.Signature | None:
    try:
        return inspect.signature(handler)
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
