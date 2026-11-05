"""FastAPI server for shell strategies.

A strategy app is assembled from lllm's own services: every tactic resource is
mounted as a full tactic service at ``/tactics/<name>/`` (``run``, ``stream``,
``info``, and custom endpoints), and a strategy with a channel store mounts the
network store API at ``/network/``. The shell adds its own surface on top:
``/strategy``, ``/resources``, channel shortcuts, and the strategy ``/run``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder

from ..services import create_tactic_app
from .models import StrategyRunRequest, StrategyRunResponse
from .strategy import Strategy


def create_strategy_app(strategy: Strategy) -> FastAPI:
    """Expose a strategy as a FastAPI application."""

    from .. import __version__

    app = FastAPI(
        title=strategy.name,
        description=strategy.description,
        version=__version__,
    )
    app.state.shell_strategy = strategy

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
        output = await strategy.arun(request.input, context=request.context)
        return StrategyRunResponse(
            output=jsonable_encoder(output),
            strategy=strategy.name,
        )

    @app.post("/resources/{name}/invoke", response_model=StrategyRunResponse)
    async def invoke_resource(
        name: str,
        request: StrategyRunRequest,
    ) -> StrategyRunResponse:
        return await _invoke(strategy, name, request, endpoint="resource")

    for name, tactic in strategy.tactics().items():
        app.mount(f"/tactics/{name}", create_tactic_app(tactic))
    if strategy.store is not None:
        from ..network.server import create_app as create_store_app

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
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StrategyRunResponse(
        output=jsonable_encoder(output),
        strategy=strategy.name,
        metadata={"resource": name, "endpoint": endpoint},
    )
