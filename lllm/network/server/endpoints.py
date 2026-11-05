"""Custom endpoint metadata for LLLM network services."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
EndpointScope = Literal["store", "channel", "subscription", "artifact", "snapshot"]
F = TypeVar("F", bound=Callable[..., Any])

_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_ENDPOINT_SCOPES = {"store", "channel", "subscription", "artifact", "snapshot"}
_ROUTE_PARAMETER_RE = re.compile(r"\{[^{}]+\}")


@dataclass(frozen=True)
class StoreEndpointSpec:
    method: HttpMethod
    path: str
    name: str
    scope: EndpointScope = "store"
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", _http_method(self.method))
        object.__setattr__(self, "path", _endpoint_path(self.path))
        object.__setattr__(self, "name", _metadata_name(self.name, "endpoint name"))
        object.__setattr__(self, "scope", _endpoint_scope(self.scope))
        object.__setattr__(self, "description", _description(self.description))
        object.__setattr__(self, "tags", _tags(self.tags))


class endpoint:
    """Decorator namespace for custom network store/channel endpoints.

    ``create_app`` binds handler parameters by name: ``store`` receives the
    store, ``body`` the parsed JSON body (``None`` for GET, DELETE, or a body
    that is not JSON), ``params`` the query parameters as a ``dict`` of strings
    (the last value wins for repeated keys), and any other parameter the path
    parameter of the same name.
    """

    @staticmethod
    def get(
        path: str,
        *,
        name: str | None = None,
        scope: EndpointScope = "store",
        description: str = "",
        tags: Sequence[str] = (),
    ) -> Callable[[F], F]:
        return _attach(
            "GET",
            path,
            name=name,
            scope=scope,
            description=description,
            tags=tags,
        )

    @staticmethod
    def post(
        path: str,
        *,
        name: str | None = None,
        scope: EndpointScope = "store",
        description: str = "",
        tags: Sequence[str] = (),
    ) -> Callable[[F], F]:
        return _attach(
            "POST",
            path,
            name=name,
            scope=scope,
            description=description,
            tags=tags,
        )

    @staticmethod
    def put(
        path: str,
        *,
        name: str | None = None,
        scope: EndpointScope = "store",
        description: str = "",
        tags: Sequence[str] = (),
    ) -> Callable[[F], F]:
        return _attach(
            "PUT",
            path,
            name=name,
            scope=scope,
            description=description,
            tags=tags,
        )

    @staticmethod
    def patch(
        path: str,
        *,
        name: str | None = None,
        scope: EndpointScope = "store",
        description: str = "",
        tags: Sequence[str] = (),
    ) -> Callable[[F], F]:
        return _attach(
            "PATCH",
            path,
            name=name,
            scope=scope,
            description=description,
            tags=tags,
        )

    @staticmethod
    def delete(
        path: str,
        *,
        name: str | None = None,
        scope: EndpointScope = "store",
        description: str = "",
        tags: Sequence[str] = (),
    ) -> Callable[[F], F]:
        return _attach(
            "DELETE",
            path,
            name=name,
            scope=scope,
            description=description,
            tags=tags,
        )


def endpoint_spec(fn: Callable[..., Any]) -> StoreEndpointSpec | None:
    spec = getattr(fn, "__lllm_network_endpoint__", None)
    return spec if isinstance(spec, StoreEndpointSpec) else None


def endpoint_specs(
    functions: Sequence[Callable[..., Any]],
) -> tuple[tuple[Callable[..., Any], StoreEndpointSpec], ...]:
    endpoints = tuple(
        (fn, spec) for fn in functions if (spec := endpoint_spec(fn)) is not None
    )
    _validate_endpoint_collection([spec for _, spec in endpoints])
    return endpoints


def endpoint_route_key(spec: StoreEndpointSpec) -> tuple[str, str]:
    return spec.method, endpoint_path_key(spec.path)


def endpoint_path_key(path: str) -> str:
    return _ROUTE_PARAMETER_RE.sub("{}", path)


def _attach(
    method: HttpMethod,
    path: str,
    *,
    name: str | None,
    scope: EndpointScope,
    description: str,
    tags: Sequence[str],
) -> Callable[[F], F]:
    normalized = _endpoint_path(path)
    endpoint_name = None if name is None else _metadata_name(name, "endpoint name")
    endpoint_scope = _endpoint_scope(scope)
    endpoint_description = _description(description)
    endpoint_tags = _tags(tags)

    def decorator(fn: F) -> F:
        setattr(
            fn,
            "__lllm_network_endpoint__",
            StoreEndpointSpec(
                method=method,
                path=normalized,
                name=endpoint_name or fn.__name__,
                scope=endpoint_scope,
                description=endpoint_description or (fn.__doc__ or "").strip(),
                tags=endpoint_tags,
            ),
        )
        return fn

    return decorator


def _http_method(method: str) -> HttpMethod:
    if not isinstance(method, str) or not method.strip():
        raise ValueError("endpoint method must be a non-empty HTTP method")
    if any(ch.isspace() for ch in method):
        raise ValueError("endpoint method must not contain whitespace")
    value = method.upper()
    if value not in _HTTP_METHODS:
        raise ValueError(f"unsupported endpoint method: {method!r}")
    return value  # type: ignore[return-value]


def _endpoint_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("endpoint path must be a non-empty route path")
    if "%" in path:
        raise ValueError("endpoint path must not contain percent escapes")
    if any(ch.isspace() for ch in path):
        raise ValueError("endpoint path must not contain whitespace")
    if path.startswith("//"):
        raise ValueError("endpoint path must not be a network-path reference")
    if "://" in path or "?" in path or "#" in path:
        raise ValueError("endpoint path must be a route path, not a URL or query")
    return path if path.startswith("/") else f"/{path}"


def _metadata_name(name: str, label: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label} must be a non-empty string")
    if "%" in name:
        raise ValueError(f"{label} must not contain percent escapes")
    if any(ch.isspace() for ch in name):
        raise ValueError(f"{label} must not contain whitespace")
    return name


def _endpoint_scope(scope: str) -> EndpointScope:
    if not isinstance(scope, str) or scope not in _ENDPOINT_SCOPES:
        raise ValueError(f"unsupported endpoint scope: {scope!r}")
    return scope  # type: ignore[return-value]


def _description(description: str) -> str:
    if not isinstance(description, str):
        raise ValueError("endpoint description must be a string")
    return description.strip()


def _tags(tags: Sequence[str]) -> tuple[str, ...]:
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        raise ValueError("endpoint tags must be a sequence of strings")
    values: list[str] = []
    for tag in tags:
        values.append(_metadata_name(tag, "endpoint tag"))
    return tuple(values)


def _validate_endpoint_collection(specs: Sequence[StoreEndpointSpec]) -> None:
    names: set[str] = set()
    routes: dict[tuple[str, str], StoreEndpointSpec] = {}
    for spec in specs:
        if spec.name in names:
            raise ValueError(f"duplicate custom endpoint name: {spec.name!r}")
        names.add(spec.name)

        key = endpoint_route_key(spec)
        if key in routes:
            raise ValueError(
                "duplicate custom endpoint route: "
                f"{spec.method} {spec.path!r}"
            )
        routes[key] = spec
