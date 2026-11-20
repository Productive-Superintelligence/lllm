"""FastAPI service adapter for tactics."""

import copy
import inspect
from contextlib import aclosing
import json
import logging
import re
import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    Field,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from ..protocol import (
    CallContext,
    LLLMError,
    OutputSchemaError,
    SchemaError,
    Tactic,
    TacticEvent,
    TacticUnsupportedError,
)
from ..protocol._validation import (
    copy_boundary_value,
    mapping_field_value,
    metadata_field_value,
    path_segment_value,
    public_boundary_value,
    token_value,
)
from .endpoints import (
    EndpointSpec,
    custom_endpoints,
    endpoint_path_key,
    endpoint_requires_body,
    endpoint_route_key,
)

logger = logging.getLogger(__name__)

_JSON_SCHEMA_DEFS_REF = "#/$defs/"
_OPENAPI_SCHEMAS_REF = "#/components/schemas/"


class CrossSiteRequestError(LLLMError):
    """Raised when a browser sends a tactic route a request from another site
    that browsers send without asking first.

    A web page can make a visitor's browser send a form, plain text, or a
    body without a content type, and a plain ``GET``, to any address, a local
    service included, with no CORS preflight. Services refuse such a
    cross-site ``POST`` that is not JSON, and a cross-site ``GET`` or ``HEAD``
    to a custom endpoint, with status 403. ``Sec-Fetch-Site`` decides whether
    a request is cross-site when present (``cross-site`` and ``same-site``
    are), and otherwise its ``Origin``: ``null``, an origin that is not
    ``http`` or ``https``, or another host and port than ``Host``.
    """


class RunRequest(BaseModel):
    """Canonical request envelope for tactic calls."""

    input: Any = None
    task: Any = None
    context: dict[str, Any] | None = None

    @property
    def value(self) -> Any:
        return self.input if self.input is not None else self.task

    def model_post_init(self, __context: Any) -> None:
        self.input = copy_boundary_value(self.input)
        self.task = copy_boundary_value(self.task)
        self.context = copy_boundary_value(self.context)

    @field_validator("context", mode="before")
    @classmethod
    def _validate_context(cls, value: Any) -> Any:
        if value is None:
            return value
        return mapping_field_value("context", value)


class RunResponse(BaseModel):
    """Canonical response envelope for successful tactic calls."""

    output: Any = None
    request_id: StrictStr
    tactic: StrictStr

    def model_post_init(self, __context: Any) -> None:
        self.output = copy_boundary_value(self.output)

    @model_validator(mode="after")
    def _validate_request_id(self) -> "RunResponse":
        token_value(self.request_id, "request_id")
        path_segment_value(self.tactic, "tactic")
        return self


class ErrorDetail(BaseModel):
    """Stable service error body."""

    type: StrictStr
    message: StrictStr
    tactic: StrictStr | None = None
    endpoint: StrictStr | None = None
    request_id: StrictStr | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.metadata = copy_boundary_value(self.metadata)

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: Any) -> Any:
        return metadata_field_value("metadata", value)

    @model_validator(mode="after")
    def _validate_identity(self) -> "ErrorDetail":
        token_value(self.type, "error.type")
        if self.tactic is not None:
            path_segment_value(self.tactic, "tactic")
        if self.endpoint is not None:
            token_value(self.endpoint, "endpoint")
        if self.request_id is not None:
            token_value(self.request_id, "request_id")
        return self


class ErrorResponse(BaseModel):
    error: ErrorDetail


def create_tactic_app(
    tactic: Tactic[Any, Any],
    *,
    title: str | None = None,
    description: str | None = None,
    allow_cross_site_requests: bool = False,
):
    """Create a FastAPI app for one tactic.

    ``allow_cross_site_requests`` is as for :func:`create_service_app`.
    """

    return create_service_app(
        {tactic.tactic_name: tactic},
        title=title or tactic.info().name,
        description=description or tactic.info().description,
        expose_single_tactic_routes=True,
        allow_cross_site_requests=allow_cross_site_requests,
    )


def create_service_app(
    tactics: Mapping[str, Tactic[Any, Any]] | Sequence[Tactic[Any, Any]],
    *,
    title: str = "LLLM Tactic Service",
    description: str = "",
    expose_single_tactic_routes: bool = False,
    allow_cross_site_requests: bool = False,
):
    """Create a FastAPI app exposing one or more tactics.

    The routes that run a tactic refuse the requests a browser sends from
    another site without asking first: a ``POST`` that is not JSON, and a
    ``GET`` or ``HEAD`` to a custom endpoint (see
    :class:`CrossSiteRequestError`). An app that serves browsers on other
    origins, with its own CORS settings, passes
    ``allow_cross_site_requests=True``.
    """

    expose_single_tactic_routes_value = _bool_value(
        "expose_single_tactic_routes",
        expose_single_tactic_routes,
    )
    check_cross_site = not _bool_value(
        "allow_cross_site_requests",
        allow_cross_site_requests,
    )
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import StreamingResponse
        from fastapi.encoders import jsonable_encoder
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install lllm-core[server] to use FastAPI services.") from exc

    tactic_map = _normalize_tactics(tactics)
    from .. import __version__

    app = FastAPI(title=title, description=description, version=__version__)
    app.state.lllm_tactics = tactic_map

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "tactics": sorted(tactic_map)}

    @app.get("/tactics")
    async def list_tactics() -> list[dict[str, Any]]:
        return [_public_tactic_info(tactic, jsonable_encoder) for tactic in tactic_map.values()]

    @app.get("/tactics/{name}/info")
    async def tactic_info(name: str) -> dict[str, Any]:
        tactic = _get_tactic(name, tactic_map)
        return _public_tactic_info(tactic, jsonable_encoder)

    @app.post("/tactics/{name}/run", response_model=RunResponse)
    async def run_tactic(name: str, request: Request) -> RunResponse:
        tactic = _get_tactic(name, tactic_map)
        context = _error_context(tactic, "run")
        try:
            if check_cross_site:
                _refuse_cross_site_request(request)
            input_value, context = await _input_and_context_from_request(
                request,
                tactic=tactic,
                endpoint="run",
            )
            output = await tactic.arun(input_value, context=context)
            return _run_response(output, tactic=tactic, context=context)
        except Exception as exc:
            raise _http_error(exc, tactic=tactic, endpoint="run", context=context) from exc

    @app.post("/tactics/{name}/stream")
    async def stream_tactic(name: str, request: Request):
        tactic = _get_tactic(name, tactic_map)
        context = _error_context(tactic, "stream")
        try:
            if check_cross_site:
                _refuse_cross_site_request(request)
            input_value, context = await _input_and_context_from_request(
                request,
                tactic=tactic,
                endpoint="stream",
            )
            if not tactic.supports("stream") and not tactic.supports("events"):
                raise TacticUnsupportedError(
                    f"Tactic '{name}' does not support streaming."
                )
            result = tactic.aevents(input_value, context=context)
            return StreamingResponse(
                _event_chunks(
                    result,
                    tactic=tactic,
                    endpoint="stream",
                    context=context,
                ),
                media_type="text/event-stream",
            )
        except Exception as exc:
            raise _http_error(exc, tactic=tactic, endpoint="stream", context=context) from exc

    if len(tactic_map) == 1 and expose_single_tactic_routes_value:
        only = next(iter(tactic_map.values()))

        @app.get("/info")
        async def single_info() -> dict[str, Any]:
            return _public_tactic_info(only, jsonable_encoder)

        @app.post("/run", response_model=RunResponse)
        async def single_run(request: Request) -> RunResponse:
            context = _error_context(only, "run")
            try:
                if check_cross_site:
                    _refuse_cross_site_request(request)
                input_value, context = await _input_and_context_from_request(
                    request,
                    tactic=only,
                    endpoint="run",
                )
                output = await only.arun(input_value, context=context)
                return _run_response(output, tactic=only, context=context)
            except Exception as exc:
                raise _http_error(exc, tactic=only, endpoint="run", context=context) from exc

        @app.post("/stream")
        async def single_stream(request: Request):
            context = _error_context(only, "stream")
            try:
                if check_cross_site:
                    _refuse_cross_site_request(request)
                input_value, context = await _input_and_context_from_request(
                    request,
                    tactic=only,
                    endpoint="stream",
                )
                if not only.supports("stream") and not only.supports("events"):
                    raise TacticUnsupportedError(
                        f"Tactic '{only.tactic_name}' does not support streaming."
                    )
                result = only.aevents(input_value, context=context)
                return StreamingResponse(
                    _event_chunks(
                        result,
                        tactic=only,
                        endpoint="stream",
                        context=context,
                    ),
                    media_type="text/event-stream",
                )
            except Exception as exc:
                raise _http_error(
                    exc,
                    tactic=only,
                    endpoint="stream",
                    context=context,
                ) from exc

        _document_tactic_schemas(app, only, jsonable_encoder)

    reserved_routes = _reserved_routes(
        tactic_map,
        expose_single_tactic_routes=expose_single_tactic_routes_value,
    )
    seen_custom_routes: dict[tuple[str, str], str] = {}
    for tactic in tactic_map.values():
        _mount_custom_endpoints(
            app,
            tactic,
            reserved_routes=reserved_routes,
            seen_custom_routes=seen_custom_routes,
            check_cross_site=check_cross_site,
        )

    return app


def _mount_custom_endpoints(
    app: Any,
    tactic: Tactic[Any, Any],
    *,
    reserved_routes: set[tuple[str, str]],
    seen_custom_routes: dict[tuple[str, str], str],
    check_cross_site: bool = True,
) -> None:
    try:
        from fastapi.responses import StreamingResponse
        from fastapi.encoders import jsonable_encoder
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install lllm-core[server] to use FastAPI services.") from exc

    for spec, method in custom_endpoints(tactic):
        route_key = endpoint_route_key(spec)
        route_label = f"{spec.method} {spec.path}"
        owner = f"{tactic.tactic_name}.{spec.name}"
        _validate_custom_route(
            route_key,
            route_label=route_label,
            owner=owner,
            reserved_routes=reserved_routes,
            seen_custom_routes=seen_custom_routes,
        )
        route = _make_custom_route(
            tactic=tactic,
            spec=spec,
            method=method,
            jsonable_encoder=jsonable_encoder,
            streaming_response=StreamingResponse,
            check_cross_site=check_cross_site,
        )
        route.__name__ = _route_name(f"{tactic.tactic_name}_{spec.name}")
        route.__doc__ = spec.description
        app.add_api_route(
            spec.path,
            route,
            methods=[spec.method],
            summary=spec.name.replace("_", " ").title(),
            description=spec.description,
            tags=list(spec.tags),
        )
        seen_custom_routes[route_key] = owner


def _reserved_routes(
    tactics: Mapping[str, Tactic[Any, Any]],
    *,
    expose_single_tactic_routes: bool,
) -> set[tuple[str, str]]:
    routes = {
        _route_key("GET", "/health"),
        _route_key("GET", "/tactics"),
        _route_key("GET", "/tactics/{name}/info"),
        _route_key("POST", "/tactics/{name}/run"),
        _route_key("POST", "/tactics/{name}/stream"),
    }
    for name in tactics:
        routes.add(_route_key("GET", f"/tactics/{name}/info"))
        routes.add(_route_key("POST", f"/tactics/{name}/run"))
        routes.add(_route_key("POST", f"/tactics/{name}/stream"))
    if expose_single_tactic_routes and len(tactics) == 1:
        routes.update(
            {
                _route_key("GET", "/info"),
                _route_key("POST", "/run"),
                _route_key("POST", "/stream"),
            }
        )
    return routes


def _route_key(method: str, path: str) -> tuple[str, str]:
    return method, endpoint_path_key(path)


def _validate_custom_route(
    route_key: tuple[str, str],
    *,
    route_label: str,
    owner: str,
    reserved_routes: set[tuple[str, str]],
    seen_custom_routes: dict[tuple[str, str], str],
) -> None:
    method, path = route_key
    if route_key in reserved_routes or _matches_reserved_tactic_route(method, path):
        raise ValueError(
            f"Custom endpoint route {route_label} for {owner} "
            "conflicts with a reserved LLLM service route."
        )
    if route_key in seen_custom_routes:
        raise ValueError(
            f"Duplicate custom endpoint route {route_label}: "
            f"{seen_custom_routes[route_key]} and {owner}"
        )


def _matches_reserved_tactic_route(method: str, path: str) -> bool:
    if method == "GET":
        return bool(re.fullmatch(r"/tactics/[^/]+/info", path))
    if method == "POST":
        return bool(re.fullmatch(r"/tactics/[^/]+/(run|stream)", path))
    return False


def _make_custom_route(
    *,
    tactic: Tactic[Any, Any],
    spec: EndpointSpec,
    method: Any,
    jsonable_encoder: Any,
    streaming_response: Any,
    check_cross_site: bool = True,
):
    from fastapi import Request
    from fastapi.responses import JSONResponse

    async def route(request: Request):
        context = _error_context(tactic, spec.name)
        try:
            if check_cross_site:
                _refuse_cross_site_request(request)
            if endpoint_requires_body(spec) or await request.body():
                input_value, context = await _input_and_context_from_request(
                    request,
                    tactic=tactic,
                    endpoint=spec.name,
                )
                input_value = tactic.validate_input(input_value)
            else:
                input_value = _query_input(request.query_params)
                context = call_context_from_data(None, tactic=tactic, endpoint=spec.name)
            if spec.mode in {"stream", "events"}:
                result = _invoke_custom(method, input_value, context=context)
                return streaming_response(
                    _event_chunks(
                        _aiter_custom_result(result),
                        tactic=tactic,
                        endpoint=spec.name,
                        context=context,
                    ),
                    media_type="text/event-stream",
                )
            result = _invoke_custom(method, input_value, context=context)
            if inspect.isawaitable(result):
                result = await result
            # Rendered here, so a result that cannot be encoded is this
            # route's error rather than a bare 500.
            return JSONResponse(jsonable_encoder(result))
        except Exception as exc:
            raise _http_error(exc, tactic=tactic, endpoint=spec.name, context=context) from exc

    return route


def _normalize_tactics(
    tactics: Mapping[str, Tactic[Any, Any]] | Sequence[Tactic[Any, Any]],
) -> dict[str, Tactic[Any, Any]]:
    normalized: dict[str, Tactic[Any, Any]] = {}
    if isinstance(tactics, Mapping):
        for name, tactic in tactics.items():
            _add_tactic(normalized, name, tactic)
        if not normalized:
            raise ValueError("create_service_app requires at least one tactic.")
        return normalized
    for tactic in tactics:
        _add_tactic(normalized, _require_tactic(tactic).tactic_name, tactic)
    if not normalized:
        raise ValueError("create_service_app requires at least one tactic.")
    return normalized


def _add_tactic(
    tactics: dict[str, Tactic[Any, Any]],
    name: Any,
    tactic: Tactic[Any, Any],
) -> None:
    tactic = _require_tactic(tactic)
    _require_route_segment("tactic.name", name)
    if name in tactics:
        raise ValueError(f"Duplicate tactic.name route: {name}")
    tactics[name] = tactic


def _require_tactic(value: Any) -> Tactic[Any, Any]:
    if not isinstance(value, Tactic):
        raise ValueError("create_service_app tactics must be Tactic instances.")
    return value


def _require_route_segment(field_name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "%" in value
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\;")
    ):
        raise ValueError(f"{field_name} must be a non-empty path segment.")


def _get_tactic(name: str, tactics: Mapping[str, Tactic[Any, Any]]) -> Tactic[Any, Any]:
    try:
        return tactics[name]
    except KeyError as exc:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error=ErrorDetail(
                    type="TacticNotFound",
                    message=_encodable_text(f"Tactic not found: {name}"),
                    tactic=_encodable_text(name),
                )
            ).model_dump(mode="json"),
        ) from exc


def _public_tactic_info(tactic: Tactic[Any, Any], jsonable_encoder: Any) -> dict[str, Any]:
    payload = jsonable_encoder(tactic.info())
    payload["examples"] = public_boundary_value(payload.get("examples", []))
    payload["metadata"] = public_boundary_value(payload.get("metadata", {}))
    return payload


def _document_tactic_schemas(
    app: Any,
    tactic: Tactic[Any, Any],
    jsonable_encoder: Any,
) -> None:
    """Show the tactic's input/output schemas on ``/run`` and ``/stream`` in OpenAPI.

    These routes read the raw request (protocol envelope or raw JSON, with LLLM
    error envelopes), so FastAPI cannot infer their bodies from typed
    parameters. Instead, each generated document is patched once with the
    schemas that ``tactic.info()`` advertises on ``/info``, encoded the same
    way. This is best effort: if ``info()`` fails or its schemas cannot be
    rendered as JSON, the document is served without them and patching is
    retried on the next call. A warning is logged when patching starts
    failing; retries that fail again are logged at debug level.
    """

    generate_openapi = app.openapi
    lock = threading.Lock()
    documented: dict[str, Any] | None = None
    failing = False

    def openapi() -> dict[str, Any]:
        nonlocal documented, failing
        with lock:
            document = generate_openapi()
            if document is not documented:
                try:
                    _add_tactic_schemas(document, tactic, jsonable_encoder)
                except Exception:
                    logger.log(
                        logging.DEBUG if failing else logging.WARNING,
                        "Could not add the schemas of tactic %r to the OpenAPI document.",
                        tactic.tactic_name,
                        exc_info=True,
                    )
                    failing = True
                    return document
                documented, failing = document, False
            return document

    app.openapi = openapi


def _add_tactic_schemas(
    document: dict[str, Any],
    tactic: Tactic[Any, Any],
    jsonable_encoder: Any,
) -> None:
    """Patch *document* in place, leaving it untouched if anything fails."""

    # Encoded as on /info: inf and nan become null; UUIDs, dates and decimals
    # become strings.
    info = jsonable_encoder(tactic.info(), include={"input_schema", "output_schema"})
    run = document["paths"]["/run"]["post"]
    stream = document["paths"]["/stream"]["post"]
    components = dict(document.get("components", {}).get("schemas", {}))
    input_schema, output_schema = _openapi_schemas(
        [info["input_schema"], info["output_schema"]],
        components,
    )
    request_schema = RunRequest.model_json_schema()
    request_schema["properties"]["input"] = input_schema
    response_schema = RunResponse.model_json_schema(mode="serialization")
    response_schema["properties"]["output"] = output_schema
    # /openapi.json is rendered as strict UTF-8 JSON (Starlette's JSONResponse),
    # so anything it would reject must fail here, before the document changes.
    json.dumps(
        [components, request_schema, response_schema],
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    document.setdefault("components", {})["schemas"] = components
    for operation in (run, stream):
        operation["requestBody"] = {
            "required": True,
            "content": {"application/json": {"schema": copy.deepcopy(request_schema)}},
        }
    responses = run.setdefault("responses", {})
    responses.setdefault("200", {}).setdefault("content", {})["application/json"] = {
        "schema": response_schema,
    }


def _openapi_schemas(
    schemas: list[dict[str, Any] | None],
    components: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return *schemas* with their local ``$defs`` moved into OpenAPI components.

    JSON Schema refs such as ``#/$defs/Item`` do not resolve once a schema is
    embedded in an OpenAPI document, so definitions become component schemas
    (renamed when a name is already taken) and refs, including discriminator
    mapping targets, are rewritten to match.
    """

    definitions: dict[str, Any] = {}
    for schema in schemas:
        for name, definition in ((schema or {}).get("$defs") or {}).items():
            if definitions.setdefault(name, definition) != definition:
                # One name, two different models: name each schema's defs apart.
                return [_openapi_schemas([item], components)[0] for item in schemas]
    names: dict[str, str] = {}
    for name in definitions:
        target, index = name, 1
        while target in components or target in names.values():
            index += 1
            target = f"{name}{index}"
        names[name] = target
    for name, definition in definitions.items():
        components[names[name]] = _openapi_refs(definition, names)
    hoisted = [_openapi_refs(schema or {}, names) for schema in schemas]
    for schema in hoisted:
        schema.pop("$defs", None)
    return hoisted


def _openapi_refs(value: Any, names: Mapping[str, str]) -> Any:
    if isinstance(value, list):
        return [_openapi_refs(item, names) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _openapi_refs(item, names) for key, item in value.items()}
    if isinstance(result.get("$ref"), str):
        result["$ref"] = _openapi_ref(result["$ref"], names)
    discriminator = result.get("discriminator")
    if isinstance(discriminator, dict) and isinstance(discriminator.get("mapping"), dict):
        discriminator["mapping"] = {
            key: _openapi_ref(target, names) if isinstance(target, str) else target
            for key, target in discriminator["mapping"].items()
        }
    return result


def _openapi_ref(ref: str, names: Mapping[str, str]) -> str:
    if not ref.startswith(_JSON_SCHEMA_DEFS_REF):
        return ref
    name = ref.removeprefix(_JSON_SCHEMA_DEFS_REF)
    return _OPENAPI_SCHEMAS_REF + names.get(name, name)


def _refuse_cross_site_request(request: Any) -> None:
    """Raise ``CrossSiteRequestError`` for a request a browser may send from
    another site without asking first.

    That is a cross-site ``POST`` whose content type is not JSON, or a
    cross-site ``GET`` or ``HEAD``, which only custom endpoints check.
    Browsers preflight ``PUT``, ``PATCH``, and ``DELETE`` across origins, so
    the app's CORS settings decide those, and they are never refused here.
    """

    method = request.method.upper()
    if method == "POST":
        if _is_json_media_type(request.headers.get("content-type")):
            return
        message = (
            "Refused a cross-site browser POST that is not JSON: any website "
            "can make a browser send forms and text here. Send JSON "
            "(Content-Type: application/json)."
        )
    elif method in {"GET", "HEAD"}:
        message = (
            f"Refused a cross-site browser {method} to a custom endpoint: any "
            "website can make a browser send one. Call it from the app's own "
            "origin, or create the app with allow_cross_site_requests=True."
        )
    else:
        return
    if _is_cross_site(request.headers):
        raise CrossSiteRequestError(message)


def _is_cross_site(headers: Any) -> bool:
    """Whether a request comes from a page on another site.

    ``Sec-Fetch-Site`` decides when present, since pages cannot set it and
    it survives proxies that rewrite ``Host``: ``cross-site`` and
    ``same-site`` are cross-site, ``same-origin`` and ``none`` are not.
    Without it, an ``Origin`` decides: ``null``, an origin that is not
    ``http`` or ``https``, or another host and port than ``Host``.
    """

    fetch_site = headers.get("sec-fetch-site")
    if fetch_site is not None:
        return fetch_site.strip().lower() not in {"same-origin", "none"}
    return _is_cross_origin(headers.get("origin"), headers.get("host"))


def _is_json_media_type(content_type: str | None) -> bool:
    media_type = _media_type(content_type)
    return media_type == "application/json" or media_type.endswith("+json")


def _media_type(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def _is_cross_origin(origin: str | None, host: str | None) -> bool:
    if origin is None:
        return False
    try:
        source = urlsplit(origin)
        target = urlsplit(f"//{host}") if host else None
        if (
            source.scheme not in {"http", "https"}
            or not source.hostname
            or target is None
            or not target.hostname
        ):
            return True  # Origin "null", or nothing to compare it with
        # A missing port is the default port of the scheme the page used,
        # which a proxy that terminates TLS hides from the app.
        default = 443 if source.scheme == "https" else 80
        return (source.hostname, source.port or default) != (
            target.hostname,
            target.port or default,
        )
    except ValueError:  # a malformed host or port
        return True


async def _input_and_context_from_request(
    request: Any,
    *,
    tactic: Tactic[Any, Any],
    endpoint: str,
) -> tuple[Any, CallContext]:
    body = _body_value(await request.body(), request.headers.get("content-type"))

    value = body
    context_data: Mapping[str, Any] | None = None
    if isinstance(body, Mapping) and _is_protocol_envelope(body):
        try:
            request_model = RunRequest.model_validate(body)
        except Exception as exc:
            raise SchemaError(f"Invalid request envelope: {exc}") from exc
        value = request_model.value
        context_data = request_model.context

    context = call_context_from_data(
        context_data,
        tactic=tactic,
        endpoint=endpoint,
    )
    return value, context


def _body_value(raw: bytes, content_type: str | None) -> Any:
    """What a request body holds, before any envelope in it is read.

    The body is decoded strictly as UTF-8 and is then the JSON it parses as,
    or else the text; an empty body is None. A body that is not UTF-8 is a
    ``SchemaError``, except an ``application/octet-stream`` body, which the
    tactic gets as bytes. A lone surrogate is a ``SchemaError`` however it
    is encoded: as raw bytes it fails the decode, and as a JSON escape
    (``\\ud800``) it is found in the parsed value.
    """

    if not raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        if _media_type(content_type) == "application/octet-stream":
            return raw
        raise SchemaError("Request body must be JSON or UTF-8 text.") from exc
    try:
        value = json.loads(text.removeprefix("\ufeff"))
    except (ValueError, RecursionError):
        return text
    if "\\u" in text and _has_lone_surrogate(value):
        raise SchemaError(
            "Request body holds a lone surrogate, which is not valid Unicode."
        )
    return value


_LONE_SURROGATE = re.compile(f"[{chr(0xD800)}-{chr(0xDFFF)}]")


def _has_lone_surrogate(value: Any) -> bool:
    """Whether a parsed JSON value holds a lone surrogate in a key or string.

    ``json.loads`` joins escaped pairs into one character, so any surrogate
    left is unpaired. The value is walked with a stack rather than
    recursion, so a deeply nested body cannot exhaust the call stack.
    """

    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if _LONE_SURROGATE.search(item):
                return True
        elif isinstance(item, dict):
            stack.extend(item)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return False


def _query_input(query_params: Any) -> dict[str, Any] | None:
    values: dict[str, Any] = {}
    for key in query_params:
        items = query_params.getlist(key)
        values[key] = items[0] if len(items) == 1 else items
    return values or None


def _is_protocol_envelope(body: Mapping[str, Any]) -> bool:
    keys = set(body)
    return bool(keys & {"input", "task"}) and keys <= {"input", "task", "context"}


def call_context_from_data(
    context_data: Mapping[str, Any] | None,
    *,
    tactic: Tactic[Any, Any],
    endpoint: str,
) -> CallContext:
    """Build the ``CallContext`` a served tactic gets from a request's ``context``.

    A missing ``request_id`` (or ``call_id``) gets a fresh one, a missing
    ``tactic_ref`` defaults to the tactic's ``package_ref``, ``endpoint`` names
    the route, and unknown keys are ignored. A malformed context raises
    ``SchemaError``.
    """

    if context_data is not None and not isinstance(context_data, Mapping):
        raise SchemaError("Request context must be an object.")
    data = context_data or {}
    metadata = data.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise SchemaError("Request context metadata must be an object.")
    tags = data.get("tags")
    if tags is None:
        tags = {}
    if not isinstance(tags, Mapping):
        raise SchemaError("Request context tags must be an object.")
    if "request_id" in data:
        request_id = data["request_id"]
    elif "call_id" in data:
        request_id = data["call_id"]
    else:
        request_id = None
    if request_id is None or request_id == "":
        request_id = CallContext().request_id
    tactic_ref = data["tactic_ref"] if "tactic_ref" in data else tactic.package_ref
    if tactic_ref is None or tactic_ref == "":
        tactic_ref = tactic.package_ref
    try:
        return CallContext(
            request_id=request_id,
            caller=data.get("caller"),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            package_ref=data.get("package_ref"),
            service_ref=data.get("service_ref"),
            tactic_ref=tactic_ref,
            endpoint=endpoint,
            metadata=dict(metadata),
            tags=dict(tags),
        )
    except ValidationError as exc:
        raise SchemaError(f"Invalid request context: {exc}") from exc


_context_from_data = call_context_from_data  # the name before it was public


def _error_context(tactic: Tactic[Any, Any], endpoint: str) -> CallContext:
    return CallContext(tactic_ref=tactic.package_ref, endpoint=endpoint)


def _run_response(
    output: Any,
    *,
    tactic: Tactic[Any, Any],
    context: CallContext,
) -> Any:
    """The ``RunResponse`` of a call, rendered as JSON right away.

    Rendering inside the route's error handling turns a result that cannot
    be encoded (bytes that are not UTF-8, a lone surrogate) into an error
    envelope instead of a bare 500.
    """

    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse

    response = RunResponse(
        output=jsonable_encoder(output),
        request_id=context.request_id,
        tactic=tactic.tactic_name,
    )
    return JSONResponse(response.model_dump(mode="json"))


async def _event_chunks(
    result: Any,
    *,
    tactic: Tactic[Any, Any] | None = None,
    endpoint: str | None = None,
    context: CallContext | None = None,
):
    try:
        # Close the tactic's stream as soon as the response ends (a client
        # that disconnects included), not whenever it is garbage collected.
        async with aclosing(_aiter_custom_result(result)) as items:
            async for item in items:
                yield _sse_chunk(item)
    except Exception as exc:
        yield _sse_chunk(
            TacticEvent.error(
                str(exc),
                **_stream_error_metadata(
                    exc,
                    tactic=tactic,
                    endpoint=endpoint,
                    context=context,
                ),
            )
        )


def _sse_chunk(item: Any) -> str:
    if isinstance(item, TacticEvent):
        payload = _public_event_payload(item)
    elif isinstance(item, BaseModel):
        payload = item.model_dump(mode="json")
    else:
        payload = item
    return f"data: {json.dumps(payload, default=str)}\n\n"


def _public_event_payload(event: TacticEvent) -> dict[str, Any]:
    payload = event.model_dump(mode="json")
    payload["metadata"] = public_boundary_value(payload.get("metadata", {}))
    return payload


def _stream_error_metadata(
    exc: Exception,
    *,
    tactic: Tactic[Any, Any] | None,
    endpoint: str | None,
    context: CallContext | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"error_type": type(exc).__name__}
    if tactic is not None:
        metadata["tactic"] = tactic.tactic_name
    if endpoint is not None:
        metadata["endpoint"] = endpoint
    if context is not None:
        metadata["request_id"] = context.request_id
    return metadata


async def _aiter_custom_result(result: Any) -> AsyncIterator[Any]:
    if inspect.isawaitable(result):
        result = await result
    if hasattr(result, "__aenter__"):
        async with result as stream:
            async for item in _aiter_custom_result(stream):
                yield item
        return
    if hasattr(result, "__aiter__"):
        if hasattr(result, "aclose"):
            async with aclosing(result) as stream:
                async for item in stream:
                    yield item
        else:
            async for item in result:
                yield item
        return
    if _is_iterable(result):
        for item in result:
            yield item
        return
    yield result


def _is_iterable(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, Mapping, BaseModel)):
        return False
    return hasattr(value, "__iter__")


def _bool_value(label: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean.")
    return value


def _invoke_custom(method: Any, input_value: Any, *, context: CallContext) -> Any:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(input_value)
    kwargs: dict[str, Any] = {}
    if "context" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        kwargs["context"] = context
    return method(input_value, **kwargs)


def _http_error(
    exc: Exception,
    *,
    tactic: Tactic[Any, Any],
    endpoint: str,
    context: CallContext,
    status_code: int | None = None,
):
    from fastapi import HTTPException

    status = status_code if status_code is not None else status_code_for_error(exc)
    return HTTPException(
        status_code=status,
        detail=ErrorResponse(
            error=ErrorDetail(
                type=type(exc).__name__,
                message=_encodable_text(str(exc)),
                tactic=tactic.tactic_name,
                endpoint=endpoint,
                request_id=context.request_id,
            )
        ).model_dump(mode="json"),
    )


def _encodable_text(text: str) -> str:
    """``text`` with any lone surrogate written as an escape, so that an
    error envelope always encodes as UTF-8."""

    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def status_code_for_error(exc: Exception) -> int:
    """The HTTP status a tactic service answers with when a call raises ``exc``.

    ``SchemaError`` and ``TacticUnsupportedError`` are 400, except
    ``OutputSchemaError``: a result that fails the tactic's output schema is
    a bug in the tactic, so it is 500, like anything else the tactic raises.
    ``CrossSiteRequestError`` is 403.
    """

    if isinstance(exc, CrossSiteRequestError):
        return 403
    if isinstance(exc, OutputSchemaError):
        return 500
    if isinstance(exc, (SchemaError, TacticUnsupportedError)):
        return 400
    return 500


_status_code_for_error = status_code_for_error  # the name before it was public


def _route_name(name: str) -> str:
    value = re.sub(r"\W+", "_", name).strip("_")
    if not value:
        return "lllm_endpoint"
    if value[0].isdigit():
        return f"endpoint_{value}"
    return value
