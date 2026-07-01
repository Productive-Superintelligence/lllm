"""Transport-neutral endpoint metadata for tactics."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

EndpointMode = Literal["run", "stream", "events"]
HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]

F = TypeVar("F", bound=Callable[..., Any])

_ENDPOINT_MODES = {"run", "stream", "events"}
_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_ROUTE_PARAMETER_RE = re.compile(r"\{[^{}]+\}")


@dataclass(frozen=True)
class EndpointSpec:
    """A domain-specific route a tactic wants to expose."""

    method: HttpMethod
    path: str
    name: str
    mode: EndpointMode = "run"
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", _http_method(self.method))
        object.__setattr__(self, "path", _endpoint_path(self.path))
        object.__setattr__(self, "name", _metadata_name(self.name, "endpoint name"))
        object.__setattr__(self, "mode", _endpoint_mode(self.mode))
        object.__setattr__(self, "description", _description(self.description))
        object.__setattr__(self, "tags", _tags(self.tags))


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

    @staticmethod
    def put(
        path: str,
        *,
        name: str | None = None,
        mode: EndpointMode = "run",
        description: str = "",
        tags: Sequence[str] = (),
    ) -> Callable[[F], F]:
        return _attach_endpoint(
            "PUT",
            path,
            name=name,
            mode=mode,
            description=description,
            tags=tags,
        )

    @staticmethod
    def patch(
        path: str,
        *,
        name: str | None = None,
        mode: EndpointMode = "run",
        description: str = "",
        tags: Sequence[str] = (),
    ) -> Callable[[F], F]:
        return _attach_endpoint(
            "PATCH",
            path,
            name=name,
            mode=mode,
            description=description,
            tags=tags,
        )

    @staticmethod
    def delete(
        path: str,
        *,
        name: str | None = None,
        mode: EndpointMode = "run",
        description: str = "",
        tags: Sequence[str] = (),
    ) -> Callable[[F], F]:
        return _attach_endpoint(
            "DELETE",
            path,
            name=name,
            mode=mode,
            description=description,
            tags=tags,
        )


def custom_endpoints(obj: Any) -> list[tuple[EndpointSpec, Callable[..., Any]]]:
    """Return endpoint metadata declared on a tactic instance."""

    discovered: list[tuple[EndpointSpec, Callable[..., Any]]] = []
    for attr_name in _static_attr_names(obj):
        if attr_name.startswith("_"):
            continue
        try:
            raw_value = inspect.getattr_static(obj, attr_name)
        except AttributeError:
            continue
        spec = _endpoint_spec(raw_value)
        if isinstance(spec, EndpointSpec):
            value = getattr(obj, attr_name)
            discovered.append((spec, value))
    _validate_endpoint_collection(discovered)
    discovered.sort(key=lambda item: item[0].name)
    return discovered


def _static_attr_names(obj: Any) -> list[str]:
    names: set[str] = set()
    try:
        instance_attrs = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        instance_attrs = {}
    if isinstance(instance_attrs, dict):
        names.update(instance_attrs)
    for cls in type(obj).__mro__:
        names.update(cls.__dict__)
    return sorted(names)


def endpoint_route_key(spec: EndpointSpec) -> tuple[str, str]:
    return spec.method, endpoint_path_key(spec.path)


def endpoint_path_key(path: str) -> str:
    return _ROUTE_PARAMETER_RE.sub("{}", path)


def _endpoint_spec(value: Any) -> EndpointSpec | None:
    spec = getattr(value, "__lllm_endpoint__", None)
    if isinstance(spec, EndpointSpec):
        return spec
    descriptor_value = getattr(value, "__func__", None)
    spec = getattr(descriptor_value, "__lllm_endpoint__", None)
    return spec if isinstance(spec, EndpointSpec) else None


def _validate_endpoint_collection(
    endpoints: list[tuple[EndpointSpec, Callable[..., Any]]],
) -> None:
    names: dict[str, str] = {}
    routes: dict[tuple[str, str], str] = {}
    for spec, method in endpoints:
        owner = getattr(method, "__name__", spec.name)
        if spec.name in names:
            raise ValueError(
                f"Duplicate endpoint name {spec.name!r}: "
                f"{names[spec.name]} and {owner}"
            )
        names[spec.name] = owner
        route = endpoint_route_key(spec)
        route_label = f"{spec.method} {spec.path}"
        if route in routes:
            raise ValueError(
                f"Duplicate custom endpoint route {route_label}: "
                f"{routes[route]} and {owner}"
            )
        routes[route] = owner


def _attach_endpoint(
    method: HttpMethod,
    path: str,
    *,
    name: str | None,
    mode: EndpointMode,
    description: str,
    tags: Sequence[str],
) -> Callable[[F], F]:
    normalized = _endpoint_path(path)
    endpoint_name = None if name is None else _metadata_name(name, "endpoint name")
    endpoint_mode = _endpoint_mode(mode)
    endpoint_description = _description(description)
    endpoint_tags = _tags(tags)

    def decorator(fn: F) -> F:
        spec = EndpointSpec(
            method=method,
            path=normalized,
            name=endpoint_name or fn.__name__,
            mode=endpoint_mode,
            description=endpoint_description or (fn.__doc__ or "").strip(),
            tags=endpoint_tags,
        )
        setattr(fn, "__lllm_endpoint__", spec)
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
    if any(ch in path for ch in "\\:;"):
        raise ValueError(
            "endpoint path must not contain backslashes, colons, or path params"
        )
    normalized = path if path.startswith("/") else f"/{path}"
    if any(part in {"", ".", ".."} for part in normalized.split("/")[1:]):
        raise ValueError("endpoint path must not contain empty or dot segments")
    return normalized


def _metadata_name(name: str, label: str) -> str:
    if (
        not isinstance(name, str)
        or not name.strip()
        or name in {".", ".."}
        or "%" in name
        or any(ch.isspace() for ch in name)
        or any(ch in name for ch in "/:\\")
    ):
        raise ValueError(
            f"{label} must be a non-empty token without whitespace, "
            "percent escapes, or path separators"
        )
    return name


def _endpoint_mode(mode: str) -> EndpointMode:
    if not isinstance(mode, str) or mode not in _ENDPOINT_MODES:
        raise ValueError(f"unsupported endpoint mode: {mode!r}")
    return mode  # type: ignore[return-value]


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
