"""FastAPI server for network stores."""

from __future__ import annotations

import base64
import binascii
import inspect
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field, StrictStr, model_validator

from ..core import (
    ArtifactNotFoundError,
    Channel,
    ChannelExistsError,
    ChannelNotFoundError,
    Event,
    EventNotFoundError,
    InvalidPayloadError,
    NetworkError,
    Snapshot,
    SnapshotNotFoundError,
    SubscriptionExistsError,
    SubscriptionNotFoundError,
)
from ..core._copy import copy_boundary_value
from ..stores import LocalStore
from .endpoints import (
    StoreEndpointSpec,
    endpoint_path_key,
    endpoint_route_key,
    endpoint_specs,
)


_CUSTOM_ROUTE_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


class ErrorDetail(BaseModel):
    type: StrictStr
    message: StrictStr

    @model_validator(mode="after")
    def _validate_type(self) -> "ErrorDetail":
        _validate_token(self.type, "error.type")
        return self


class ErrorResponse(BaseModel):
    error: ErrorDetail


class SubscriptionRequest(BaseModel):
    id: StrictStr | None = None
    channel: StrictStr
    consumer: StrictStr | None = None
    batch_size: int = 100
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.filters = copy_boundary_value(self.filters)
        self.metadata = copy_boundary_value(self.metadata)


class ArtifactWriteRequest(BaseModel):
    data: StrictStr
    encoding: Literal["text", "base64"] = "text"
    channel: StrictStr | None = None
    media_type: StrictStr = "application/octet-stream"
    metadata: dict[str, Any] = Field(default_factory=dict)
    event_ids: tuple[StrictStr, ...] = Field(default_factory=tuple)

    def model_post_init(self, __context: Any) -> None:
        self.metadata = copy_boundary_value(self.metadata)

    def bytes(self) -> bytes:
        if self.encoding == "base64":
            try:
                return base64.b64decode(self.data.encode("ascii"), validate=True)
            except (binascii.Error, ValueError) as exc:
                raise InvalidPayloadError("Invalid base64 artifact data.") from exc
        return self.data.encode("utf-8")


class SnapshotWriteRequest(BaseModel):
    channel: StrictStr | None = None
    timestamp: float | None = None
    value: Any = None
    schema_ref: StrictStr | None = Field(default=None, alias="schema")
    source_event_id: StrictStr | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.value = copy_boundary_value(self.value)
        self.metadata = copy_boundary_value(self.metadata)


def create_app(
    store: LocalStore | None = None,
    *,
    custom_endpoints: Sequence[Callable[..., Any]] = (),
):
    """Create a FastAPI app exposing a store's portable API."""

    try:
        from fastapi import FastAPI
        from fastapi.encoders import jsonable_encoder
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse, Response
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install lllm-core[server] to use the FastAPI server.") from exc

    local_store = store or LocalStore()
    app = FastAPI(title="LLLM network Store", version="0.1.1")
    app.state.network_store = local_store

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Any, exc: RequestValidationError):
        return JSONResponse(
            status_code=400,
            content={
                "detail": _error_payload(
                    "InvalidPayloadError",
                    _validation_error_message(exc),
                )
            },
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.post("/channels")
    async def create_channel(channel: Channel) -> dict[str, Any]:
        try:
            return jsonable_encoder(local_store.create_channel(channel))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/channels")
    async def list_channels() -> list[dict[str, Any]]:
        return [jsonable_encoder(channel) for channel in local_store.list_channels()]

    @app.get("/channels/{name}")
    async def get_channel(name: str) -> dict[str, Any]:
        try:
            return jsonable_encoder(local_store.get_channel(name))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/events")
    async def append_event(event: Event) -> dict[str, Any]:
        try:
            return jsonable_encoder(local_store.append_event(event))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/events")
    async def query_events(
        channel: str,
        after_cursor: int = 0,
        limit: int = 100,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            events = local_store.query_events(
                channel,
                after_cursor=after_cursor,
                limit=limit,
                kind=kind,
            )
            return [jsonable_encoder(event) for event in events]
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/events/{event_id}")
    async def get_event(event_id: str) -> dict[str, Any]:
        try:
            return jsonable_encoder(local_store.get_event(event_id))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/subscriptions")
    async def create_subscription(request: SubscriptionRequest) -> dict[str, Any]:
        try:
            return jsonable_encoder(
                local_store.create_subscription(
                    request.channel,
                    subscription_id=request.id,
                    consumer=request.consumer,
                    batch_size=request.batch_size,
                    filters=request.filters,
                    metadata=request.metadata,
                )
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/subscriptions/{subscription_id}")
    async def get_subscription(subscription_id: str) -> dict[str, Any]:
        try:
            return jsonable_encoder(local_store.get_subscription(subscription_id))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/subscriptions/{subscription_id}/pull")
    async def pull_subscription(
        subscription_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        try:
            return [
                jsonable_encoder(event)
                for event in local_store.pull_subscription(subscription_id, limit=limit)
            ]
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/artifacts")
    async def write_artifact(request: ArtifactWriteRequest) -> dict[str, Any]:
        try:
            return jsonable_encoder(
                local_store.write_artifact(
                    request.bytes(),
                    channel=request.channel,
                    media_type=request.media_type,
                    metadata=request.metadata,
                    event_ids=request.event_ids,
                )
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/artifacts/{artifact_id}")
    async def read_artifact(artifact_id: str):
        try:
            artifact = local_store.get_artifact(artifact_id)
            data = local_store.read_artifact(artifact_id)
            return Response(content=data, media_type=artifact.media_type)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/artifacts/{artifact_id}/metadata")
    async def get_artifact(artifact_id: str) -> dict[str, Any]:
        try:
            return jsonable_encoder(local_store.get_artifact(artifact_id))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.put("/snapshots/{name}")
    async def put_snapshot(name: str, snapshot: SnapshotWriteRequest) -> dict[str, Any]:
        try:
            data = snapshot.model_dump(by_alias=True, exclude_none=True)
            value = Snapshot(name=name, **data)
            return jsonable_encoder(local_store.put_snapshot(value))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/snapshots/{name}")
    async def get_snapshot(name: str) -> dict[str, Any]:
        try:
            return jsonable_encoder(local_store.get_snapshot(name))
        except Exception as exc:
            raise _http_error(exc) from exc

    reserved_routes = _reserved_routes(app)
    seen_custom_routes: dict[tuple[str, str], StoreEndpointSpec] = {}
    for fn, spec in endpoint_specs(custom_endpoints):
        _validate_custom_route(
            spec,
            reserved_routes=reserved_routes,
            seen_custom_routes=seen_custom_routes,
        )
        _mount_custom_endpoint(app, local_store, spec, fn)

    return app


def _mount_custom_endpoint(
    app: Any,
    store: LocalStore,
    spec: StoreEndpointSpec,
    fn: Callable[..., Any],
) -> None:
    from fastapi import Request
    from fastapi.encoders import jsonable_encoder

    async def route(request: Request):
        try:
            body = None
            if request.method not in {"GET", "DELETE"}:
                try:
                    body = await request.json()
                except Exception:
                    body = None
            result = fn(**_custom_kwargs(fn, store=store, body=body, path=request.path_params))
            if inspect.isawaitable(result):
                result = await result
            return jsonable_encoder(result)
        except Exception as exc:
            raise _http_error(exc) from exc

    route.__name__ = _route_name(spec.name)
    route.__doc__ = spec.description
    route.__annotations__ = {"request": Request}
    app.add_api_route(
        spec.path,
        route,
        methods=[spec.method],
        summary=spec.name.replace("_", " ").title(),
        description=spec.description,
        tags=list(spec.tags),
    )


def _custom_kwargs(
    fn: Callable[..., Any],
    *,
    store: LocalStore,
    body: Any,
    path: dict[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name == "store":
            kwargs[name] = store
        elif name == "body":
            kwargs[name] = body
        elif name in path:
            kwargs[name] = path[name]
        elif parameter.default is inspect.Parameter.empty:
            raise TypeError(f"Cannot bind custom endpoint parameter: {name}")
    return kwargs


def _route_name(name: str) -> str:
    value = re.sub(r"\W+", "_", name).strip("_")
    if not value:
        return "network_endpoint"
    if value[0].isdigit():
        return f"endpoint_{value}"
    return value


def _reserved_routes(app: Any) -> set[tuple[str, str]]:
    routes: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path_format", None) or getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if not isinstance(path, str) or not methods:
            continue
        for method in methods:
            method_name = str(method).upper()
            if method_name in _CUSTOM_ROUTE_METHODS:
                routes.add((method_name, endpoint_path_key(path)))
    return routes


def _validate_custom_route(
    spec: StoreEndpointSpec,
    *,
    reserved_routes: set[tuple[str, str]],
    seen_custom_routes: dict[tuple[str, str], StoreEndpointSpec],
) -> None:
    key = endpoint_route_key(spec)
    if key in reserved_routes:
        raise ValueError(
            "custom endpoint route conflicts with reserved LLLM network service route: "
            f"{spec.method} {spec.path}"
        )
    if key in seen_custom_routes:
        raise ValueError(
            "duplicate custom endpoint route: "
            f"{spec.method} {spec.path!r}"
        )
    seen_custom_routes[key] = spec


def _http_error(exc: Exception):
    try:
        from fastapi import HTTPException
    except ImportError as import_exc:  # pragma: no cover
        raise RuntimeError("Install lllm-core[server] to use the FastAPI server.") from import_exc

    status_code = 500
    if isinstance(
        exc,
        (
            ChannelNotFoundError,
            EventNotFoundError,
            ArtifactNotFoundError,
            SnapshotNotFoundError,
            SubscriptionNotFoundError,
        ),
    ):
        status_code = 404
    elif isinstance(exc, (ChannelExistsError, SubscriptionExistsError)):
        status_code = 409
    elif isinstance(exc, NetworkError):
        status_code = 400
    return HTTPException(
        status_code=status_code,
        detail=_error_payload(type(exc).__name__, str(exc)),
    )


def _error_payload(error_type: str, message: str) -> dict[str, Any]:
    return ErrorResponse(
        error=ErrorDetail(type=error_type, message=message)
    ).model_dump(mode="json")


def _validation_error_message(exc: Exception) -> str:
    errors = exc.errors() if hasattr(exc, "errors") else []
    messages: list[str] = []
    for error in errors:
        location = ".".join(
            str(part) for part in error.get("loc", ()) if part != "body"
        )
        message = str(error.get("msg") or "Invalid request payload.")
        messages.append(f"{location}: {message}" if location else message)
    return "; ".join(messages) if messages else "Invalid request payload."


def _validate_token(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "%" in value
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\")
    ):
        raise ValueError(f"{field_name} must be a non-empty token.")
