"""Transport-neutral endpoint metadata for tactics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

EndpointMode = Literal["run", "stream", "events"]
HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class EndpointSpec:
    """A domain-specific route a tactic wants to expose."""

    method: HttpMethod
    path: str
    name: str
    mode: EndpointMode = "run"
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


class endpoint:
    """Decorator namespace for custom tactic endpoint metadata."""

    @staticmethod
    def post(
        path: str,
        *,
        name: str | None = None,
        mode: EndpointMode = "run",
        description: str = "",
        tags: Sequence[str] = (),
    ) -> Callable[[F], F]:
        return _attach_endpoint(
            "POST",
            path,
            name=name,
            mode=mode,
            description=description,
            tags=tags,
        )

    @staticmethod
    def get(
        path: str,
        *,
        name: str | None = None,
        mode: EndpointMode = "run",
        description: str = "",
        tags: Sequence[str] = (),
    ) -> Callable[[F], F]:
        return _attach_endpoint(
            "GET",
            path,
            name=name,
            mode=mode,
            description=description,
            tags=tags,
        )


def custom_endpoints(obj: Any) -> list[tuple[EndpointSpec, Callable[..., Any]]]:
    """Return endpoint metadata declared on a tactic instance."""

    discovered: list[tuple[EndpointSpec, Callable[..., Any]]] = []
    for attr_name in dir(obj):
        if attr_name.startswith("_"):
            continue
        value = getattr(obj, attr_name)
        spec = getattr(value, "__lllm_endpoint__", None)
        if isinstance(spec, EndpointSpec):
            discovered.append((spec, value))
    discovered.sort(key=lambda item: item[0].name)
    return discovered


def _attach_endpoint(
    method: HttpMethod,
    path: str,
    *,
    name: str | None,
    mode: EndpointMode,
    description: str,
    tags: Sequence[str],
) -> Callable[[F], F]:
    normalized = path if path.startswith("/") else f"/{path}"

    def decorator(fn: F) -> F:
        spec = EndpointSpec(
            method=method,
            path=normalized,
            name=name or fn.__name__,
            mode=mode,
            description=description or (fn.__doc__ or "").strip(),
            tags=tuple(tags),
        )
        setattr(fn, "__lllm_endpoint__", spec)
        return fn

    return decorator
