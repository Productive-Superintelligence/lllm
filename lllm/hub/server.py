"""FastAPI app for the local disk-backed LLLM hub."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, StrictBool, StrictStr, field_validator

from ._metadata import (
    is_public_sensitive_metadata_key as _is_sensitive_metadata_key,
    is_schema_metadata_key as _is_schema_metadata_key,
)
from .local import LocalHub, PublishValidationError, _should_ignore_publish_name
from .manifest import require_path_value
from .validator import validate_package


class _PackagePathRequest(BaseModel):
    path: StrictStr

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return require_path_value(value, "package path")


class ValidateRequest(_PackagePathRequest):
    pass


class PublishRequest(_PackagePathRequest):
    run_validation: StrictBool = Field(default=True, alias="validate")


def create_app(*, hub: LocalHub | None = None, hub_root: str | Path = ".lllm/hub"):
    """Create a local package-hub API over a disk-backed hub."""

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.encoders import jsonable_encoder
        from fastapi.responses import PlainTextResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install lllm-core[server] to use the FastAPI server.") from exc

    local_hub = hub or LocalHub(hub_root)
    app = FastAPI(title="LLLM hub Local Hub", version="0.0.1")
    app.state.hub = local_hub

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.post("/validate")
    async def validate(request: ValidateRequest) -> dict[str, Any]:
        report = validate_package(request.path)
        return report.model_dump(mode="json")

    @app.post("/publish")
    async def publish(request: PublishRequest) -> dict[str, Any]:
        try:
            record = local_hub.publish(request.path, validate=request.run_validation)
        except PublishValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail=exc.report.model_dump(mode="json"),
            ) from exc
        return _public_record(record, jsonable_encoder)

    @app.get("/packages")
    async def list_packages() -> list[dict[str, Any]]:
        return [_public_record(record, jsonable_encoder) for record in local_hub.list()]

    @app.get("/packages/{org}/{name}")
    async def metadata(org: str, name: str, version: str | None = None) -> dict[str, Any]:
        try:
            return _public_record(
                local_hub.get(f"{org}/{name}", version=version),
                jsonable_encoder,
            )
        except (KeyError, ValueError) as exc:
            raise _lookup_error(exc) from exc

    @app.get("/packages/{org}/{name}/card")
    async def card(org: str, name: str, version: str | None = None) -> str:
        try:
            return PlainTextResponse(local_hub.card(f"{org}/{name}", version=version))
        except (KeyError, ValueError) as exc:
            raise _lookup_error(exc) from exc

    @app.get("/packages/{org}/{name}/agent-card")
    async def agent_card(org: str, name: str, version: str | None = None) -> str:
        try:
            return PlainTextResponse(local_hub.agent_card(f"{org}/{name}", version=version))
        except (KeyError, ValueError) as exc:
            raise _lookup_error(exc) from exc

    @app.get("/packages/{org}/{name}/config-template")
    async def config_template(org: str, name: str, version: str | None = None) -> str:
        try:
            return PlainTextResponse(
                local_hub.config_template(f"{org}/{name}", version=version)
            )
        except (KeyError, ValueError) as exc:
            raise _lookup_error(exc) from exc

    @app.get("/packages/{org}/{name}/download")
    async def download(org: str, name: str, version: str | None = None):
        try:
            record = local_hub.get(f"{org}/{name}", version=version)
        except (KeyError, ValueError) as exc:
            raise _lookup_error(exc) from exc
        payload = _zip_folder(record.root)
        headers = {
            "content-disposition": f'attachment; filename="{record.name}-{record.version}.zip"'
        }
        return StreamingResponse(
            io.BytesIO(payload),
            media_type="application/zip",
            headers=headers,
        )

    return app


def _lookup_error(exc: Exception):
    from fastapi import HTTPException

    status_code = 400 if isinstance(exc, ValueError) else 404
    return HTTPException(status_code=status_code, detail=str(exc))


def _public_record(record: Any, jsonable_encoder: Any) -> dict[str, Any]:
    return _public_payload(jsonable_encoder(record))


def _public_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        payload: dict[str, Any] = {}
        for key, item in value.items():
            if key == "metadata" and isinstance(item, Mapping):
                payload[key] = _public_metadata(item)
            else:
                payload[key] = _public_payload(item)
        return payload
    if isinstance(value, list):
        return [_public_payload(item) for item in value]
    return value


def _public_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        payload: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            if _is_sensitive_metadata_key(key):
                continue
            if _is_schema_metadata_key(key):
                payload[key] = item
            else:
                payload[key] = _public_metadata(item)
        return payload
    if isinstance(value, list):
        return [_public_metadata(item) for item in value]
    return value


def _zip_folder(root: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if _skip_zip_path(root, path):
                continue
            if path.is_file():
                archive.write(path, path.relative_to(root))
    return buffer.getvalue()


def _skip_zip_path(root: Path, path: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if _should_ignore_publish_name(part) or current.is_symlink():
            return True
    return False
