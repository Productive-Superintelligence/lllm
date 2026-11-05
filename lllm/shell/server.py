"""FastAPI server for shell strategies.

A strategy app is assembled from lllm's own services: every tactic resource is
mounted as a full tactic service at ``/tactics/<name>/`` (``run``, ``stream``,
``info``, and custom endpoints), and a strategy with a channel store mounts the
network store API at ``/network/``. The shell adds its own surface on top:
``/strategy``, ``/resources``, channel shortcuts, and the strategy ``/run``.

The shell routes report errors in the services error envelope,
``{"detail": {"error": {"type": ..., "message": ...}}}``: an unknown resource
is a 404; invalid input (a malformed request, or a ``ValueError`` or
``SchemaError`` from a handler) is a 400; channel store errors keep the store
server's codes (404 not found, 409 conflict, 400 otherwise); anything else is
a 500.
"""

from __future__ import annotations

import sys
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..protocol import SchemaError, TacticUnsupportedError
from ..services import ErrorDetail, ErrorResponse, create_tactic_app
from .models import StrategyRunRequest, StrategyRunResponse
from .strategy import ResourceNotFoundError, Strategy


def create_strategy_app(strategy: Strategy) -> FastAPI:
    """Expose a strategy as a FastAPI application.

    Building the app serves the strategy: it opens the strategy's store, which
    creates the declared channels, and mounts it at ``/network/``.
    """

    from .. import __version__

    app = FastAPI(
        title=strategy.name,
        description=strategy.description,
        version=__version__,
    )
    app.state.shell_strategy = strategy

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        error = SchemaError(_validation_message(exc))
        return JSONResponse(status_code=400, content={"detail": _error_payload(error)})

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "strategy": strategy.name}

    @app.get("/strategy")
    async def strategy_info() -> dict[str, Any]:
        return jsonable_encoder(strategy.info())

    @app.get("/resources")
    async def resources() -> list[dict[str, Any]]:
        return jsonable_encoder(strategy.resources)

    @app.get("/packages")
    async def packages() -> list[dict[str, Any]]:
        return [
            jsonable_encoder(resource)
            for resource in strategy.resources
            if resource.kind == "package"
        ]

    @app.get("/tactics")
    async def tactics() -> list[dict[str, Any]]:
        return [
            jsonable_encoder(resource)
            for resource in strategy.resources
            if resource.kind == "tactic"
        ]

    @app.get("/channels")
    async def channels() -> list[dict[str, Any]]:
        return [
            jsonable_encoder(resource)
            for resource in strategy.resources
            if resource.kind == "channel"
        ]

    @app.get("/channels/{name}/events")
    async def query_channel(
        name: str,
        after_cursor: int = 0,
        limit: int = 100,
        kind: str | None = None,
    ) -> dict[str, Any]:
        request = StrategyRunRequest(
            input={
                "op": "query",
                "after_cursor": after_cursor,
                "limit": limit,
                "kind": kind,
            }
        )
        response = await _invoke(strategy, name, request, endpoint="channel")
        return response.model_dump()

    @app.post("/channels/{name}/events", response_model=StrategyRunResponse)
    async def append_channel_event(
        name: str,
        request: StrategyRunRequest,
    ) -> StrategyRunResponse:
        input_value = request.input if isinstance(request.input, dict) else {}
        request = StrategyRunRequest(
            input={**input_value, "op": "append"},
            context=request.context,
        )
        return await _invoke(strategy, name, request, endpoint="channel")

    @app.post("/run", response_model=StrategyRunResponse)
    async def run(request: StrategyRunRequest) -> StrategyRunResponse:
        try:
            output = await strategy.arun(request.input, context=request.context)
        except StarletteHTTPException:
            raise
        except Exception as exc:
            raise _http_error(exc, endpoint="run") from exc
        return StrategyRunResponse(
            output=jsonable_encoder(output),
            strategy=strategy.name,
        )

    @app.post("/resources/{name}/invoke", response_model=StrategyRunResponse)
    async def invoke_resource(
        name: str,
        request: StrategyRunRequest,
    ) -> StrategyRunResponse:
        """Call a resource's handler.

        A tactic resource runs as ``POST /tactics/{name}/run`` runs it: the
        ``context`` has the shape of a ``CallContext`` and the tactic receives
        one. Other handlers receive the ``context`` as a plain object.
        """

        return await _invoke(strategy, name, request, endpoint="resource")

    for name, tactic in strategy.tactics().items():
        app.mount(f"/tactics/{name}", create_tactic_app(tactic))
    if strategy.store is not None:
        from ..network.server import create_app as create_store_app

        strategy.store.open()
        # Mount the StrategyStore itself so channels declared later are created too.
        app.mount("/network", create_store_app(strategy.store))
    return app


async def _invoke(
    strategy: Strategy,
    name: str,
    request: StrategyRunRequest,
    *,
    endpoint: str,
) -> StrategyRunResponse:
    try:
        output = await strategy.invoke_resource(
            name,
            request.input,
            context=request.context,
        )
    except StarletteHTTPException:
        raise
    except Exception as exc:
        raise _http_error(exc, endpoint=endpoint, resource=name) from exc
    return StrategyRunResponse(
        output=jsonable_encoder(output),
        strategy=strategy.name,
        metadata={"resource": name, "endpoint": endpoint},
    )


def _http_error(
    exc: Exception,
    *,
    endpoint: str,
    resource: str | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=_status_code(exc),
        detail=_error_payload(
            exc,
            endpoint=endpoint,
            metadata={} if resource is None else {"resource": resource},
        ),
    )


def _error_payload(exc: Exception, **fields: Any) -> dict[str, Any]:
    return ErrorResponse(
        error=ErrorDetail(type=type(exc).__name__, message=str(exc), **fields)
    ).model_dump(mode="json")


def _status_code(exc: Exception) -> int:
    if isinstance(exc, ResourceNotFoundError):
        return 404
    # A network error exists only once lllm.network is imported; looking it up
    # here keeps strategies without channels from importing it.
    network = sys.modules.get("lllm.network.core.errors")
    if network is not None and isinstance(exc, network.NetworkError):
        if isinstance(
            exc,
            (
                network.ChannelNotFoundError,
                network.EventNotFoundError,
                network.ArtifactNotFoundError,
                network.SnapshotNotFoundError,
                network.SubscriptionNotFoundError,
            ),
        ):
            return 404
        if isinstance(exc, (network.ChannelExistsError, network.SubscriptionExistsError)):
            return 409
        return 400
    if isinstance(exc, (ValueError, SchemaError, TacticUnsupportedError)):
        return 400
    return 500


def _validation_message(exc: RequestValidationError) -> str:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        message = str(error.get("msg") or "Invalid request.")
        messages.append(f"{location}: {message}" if location else message)
    return "; ".join(messages) or "Invalid request."
