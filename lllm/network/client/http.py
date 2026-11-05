"""HTTP clients for network servers."""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

from ..core import Artifact, Channel, Event, InvalidPayloadError, Snapshot, Subscription
from ...protocol._validation import copy_boundary_value


class RemoteStoreError(RuntimeError):
    """Raised when an LLLM network HTTP server returns an error."""

    def __init__(
        self,
        status_code: int,
        *,
        error_type: str | None = None,
        message: str | None = None,
        detail: Any = None,
    ) -> None:
        self.status_code = status_code
        self.error_type = _optional_token_value(error_type, "error_type")
        self.detail = detail
        self.message = _optional_text_value(message, "message") or _detail_message(detail)
        text = f"network server returned HTTP {status_code}"
        if self.error_type:
            text += f" ({self.error_type})"
        if self.message:
            text += f": {self.message}"
        super().__init__(text)


class RemoteStore:
    """Synchronous HTTP client for the portable LLLM network API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float | None = 30.0,
        transport: Any = None,
    ) -> None:
        self.base_url = _base_url(base_url)
        self.timeout = timeout
        self.transport = transport

    def create_channel(self, channel: Channel | dict[str, Any]) -> Channel:
        data = _dump_channel(channel)
        _require_segment("channel.name", data.get("name"))
        if "metadata" in data:
            data["metadata"] = _mapping_payload("channel.metadata", data["metadata"])
        return _model_response(
            self._request("POST", "/channels", json=data),
            Channel,
            "POST /channels",
        )

    def list_channels(self) -> tuple[Channel, ...]:
        return _model_tuple_response(
            self._request("GET", "/channels"),
            Channel,
            "GET /channels",
        )

    def get_channel(self, name: str) -> Channel:
        _require_segment("channel.name", name)
        return _model_response(
            self._request("GET", f"/channels/{name}"),
            Channel,
            "GET /channels/{name}",
        )

    def append_event(self, event: Event | dict[str, Any]) -> Event:
        data = _dump_event(event)
        _require_optional_segment("event.id", data.get("id"))
        _require_segment("event.channel", data.get("channel"))
        if "kind" in data:
            _require_token("event.kind", data["kind"])
        if data.get("correlation_id") is not None:
            _require_token("event.correlation_id", data["correlation_id"])
        if "metadata" in data:
            data["metadata"] = _mapping_payload("event.metadata", data["metadata"])
        data["parent_ids"] = _segment_list(
            "event.parent_ids",
            data.get("parent_ids", ()),
        )
        return _model_response(
            self._request("POST", "/events", json=data),
            Event,
            "POST /events",
        )

    def query_events(
        self,
        channel: str,
        *,
        after_cursor: int = 0,
        limit: int = 100,
        kind: str | None = None,
    ) -> tuple[Event, ...]:
        _require_segment("channel.name", channel)
        params = {"channel": channel, "after_cursor": after_cursor, "limit": limit}
        if kind is not None:
            _require_token("event.kind", kind)
            params["kind"] = kind
        return _model_tuple_response(
            self._request("GET", "/events", params=params),
            Event,
            "GET /events",
        )

    def get_event(self, event_id: str) -> Event:
        _require_segment("event.id", event_id)
        return _model_response(
            self._request("GET", f"/events/{event_id}"),
            Event,
            "GET /events/{id}",
        )

    def create_subscription(
        self,
        channel: str,
        *,
        subscription_id: str | None = None,
        consumer: str | None = None,
        batch_size: int = 100,
        filters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Subscription:
        _require_segment("subscription.channel", channel)
        _require_optional_segment("subscription.id", subscription_id)
        _require_optional_token("subscription.consumer", consumer)
        return _model_response(
            self._request(
                "POST",
                "/subscriptions",
                json={
                    "id": subscription_id,
                    "channel": channel,
                    "consumer": consumer,
                    "batch_size": batch_size,
                    "filters": _copy_mapping("subscription.filters", filters),
                    "metadata": _copy_mapping("subscription.metadata", metadata),
                },
            ),
            Subscription,
            "POST /subscriptions",
        )

    def pull_subscription(
        self,
        subscription_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[Event, ...]:
        _require_segment("subscription.id", subscription_id)
        params = {"limit": limit} if limit is not None else None
        return _model_tuple_response(
            self._request(
                "POST",
                f"/subscriptions/{subscription_id}/pull",
                params=params,
            ),
            Event,
            "POST /subscriptions/{id}/pull",
        )

    def get_subscription(self, subscription_id: str) -> Subscription:
        _require_segment("subscription.id", subscription_id)
        return _model_response(
            self._request("GET", f"/subscriptions/{subscription_id}"),
            Subscription,
            "GET /subscriptions/{id}",
        )

    def write_artifact(
        self,
        data: bytes | str,
        *,
        channel: str | None = None,
        media_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
        event_ids: tuple[str, ...] = (),
    ) -> Artifact:
        _require_optional_segment("artifact.channel", channel)
        event_id_list = _segment_list("artifact.event_ids", event_ids)
        payload = _artifact_payload(data)
        return _model_response(
            self._request(
                "POST",
                "/artifacts",
                json={
                    "data": payload["data"],
                    "encoding": payload["encoding"],
                    "channel": channel,
                    "media_type": media_type,
                    "metadata": _copy_mapping("artifact.metadata", metadata),
                    "event_ids": event_id_list,
                },
            ),
            Artifact,
            "POST /artifacts",
        )

    def read_artifact(self, artifact_id: str) -> bytes:
        _require_segment("artifact.id", artifact_id)
        return self._request("GET", f"/artifacts/{artifact_id}").content

    def get_artifact(self, artifact_id: str) -> Artifact:
        _require_segment("artifact.id", artifact_id)
        return _model_response(
            self._request("GET", f"/artifacts/{artifact_id}/metadata"),
            Artifact,
            "GET /artifacts/{id}/metadata",
        )

    def put_snapshot(
        self,
        name: str | Snapshot | Mapping[str, Any],
        value: Any = None,
        *,
        channel: str | None = None,
        timestamp: float | None = None,
        schema: str | None = None,
        source_event_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Snapshot:
        """Write a snapshot given as a model, a dict, or a name and value.

        ``put_snapshot(snapshot)`` takes a ``Snapshot`` or a dict of snapshot
        fields; ``put_snapshot(name, value, ...)`` stores ``value`` exactly as
        given, unless it is a ``Snapshot`` whose fields are used under that
        name. Field arguments that are not ``None`` override.
        """
        name, payload = _snapshot_payload(
            name,
            value,
            channel=channel,
            timestamp=timestamp,
            schema=schema,
            source_event_id=source_event_id,
            metadata=metadata,
        )
        _require_optional_segment("snapshot.channel", payload.get("channel"))
        _require_optional_segment(
            "snapshot.source_event_id",
            payload.get("source_event_id"),
        )
        if "metadata" in payload:
            payload["metadata"] = _mapping_payload(
                "snapshot.metadata",
                payload["metadata"],
            )
        return _model_response(
            self._request(
                "PUT",
                f"/snapshots/{name}",
                json=payload,
            ),
            Snapshot,
            "PUT /snapshots/{name}",
        )

    def get_snapshot(self, name: str) -> Snapshot:
        _require_segment("snapshot.name", name)
        return _model_response(
            self._request("GET", f"/snapshots/{name}"),
            Snapshot,
            "GET /snapshots/{name}",
        )

    def _request(self, method: str, path: str, **kwargs: Any):
        import httpx

        with httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            response = client.request(method, path, **kwargs)
        _raise_for_error(response)
        return response


class AsyncRemoteStore:
    """Async HTTP client for the portable LLLM network API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float | None = 30.0,
        transport: Any = None,
    ) -> None:
        self.base_url = _base_url(base_url)
        self.timeout = timeout
        self.transport = transport

    async def create_channel(self, channel: Channel | dict[str, Any]) -> Channel:
        data = _dump_channel(channel)
        _require_segment("channel.name", data.get("name"))
        if "metadata" in data:
            data["metadata"] = _mapping_payload("channel.metadata", data["metadata"])
        return _model_response(
            await self._request("POST", "/channels", json=data),
            Channel,
            "POST /channels",
        )

    async def list_channels(self) -> tuple[Channel, ...]:
        return _model_tuple_response(
            await self._request("GET", "/channels"),
            Channel,
            "GET /channels",
        )

    async def get_channel(self, name: str) -> Channel:
        _require_segment("channel.name", name)
        return _model_response(
            await self._request("GET", f"/channels/{name}"),
            Channel,
            "GET /channels/{name}",
        )

    async def append_event(self, event: Event | dict[str, Any]) -> Event:
        data = _dump_event(event)
        _require_optional_segment("event.id", data.get("id"))
        _require_segment("event.channel", data.get("channel"))
        if "kind" in data:
            _require_token("event.kind", data["kind"])
        if data.get("correlation_id") is not None:
            _require_token("event.correlation_id", data["correlation_id"])
        if "metadata" in data:
            data["metadata"] = _mapping_payload("event.metadata", data["metadata"])
        data["parent_ids"] = _segment_list(
            "event.parent_ids",
            data.get("parent_ids", ()),
        )
        return _model_response(
            await self._request("POST", "/events", json=data),
            Event,
            "POST /events",
        )

    async def query_events(
        self,
        channel: str,
        *,
        after_cursor: int = 0,
        limit: int = 100,
        kind: str | None = None,
    ) -> tuple[Event, ...]:
        _require_segment("channel.name", channel)
        params = {"channel": channel, "after_cursor": after_cursor, "limit": limit}
        if kind is not None:
            _require_token("event.kind", kind)
            params["kind"] = kind
        return _model_tuple_response(
            await self._request("GET", "/events", params=params),
            Event,
            "GET /events",
        )

    async def get_event(self, event_id: str) -> Event:
        _require_segment("event.id", event_id)
        return _model_response(
            await self._request("GET", f"/events/{event_id}"),
            Event,
            "GET /events/{id}",
        )

    async def create_subscription(
        self,
        channel: str,
        *,
        subscription_id: str | None = None,
        consumer: str | None = None,
        batch_size: int = 100,
        filters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Subscription:
        _require_segment("subscription.channel", channel)
        _require_optional_segment("subscription.id", subscription_id)
        _require_optional_token("subscription.consumer", consumer)
        return _model_response(
            (
                await self._request(
                    "POST",
                    "/subscriptions",
                    json={
                        "id": subscription_id,
                        "channel": channel,
                        "consumer": consumer,
                        "batch_size": batch_size,
                        "filters": _copy_mapping("subscription.filters", filters),
                        "metadata": _copy_mapping("subscription.metadata", metadata),
                    },
                )
            ),
            Subscription,
            "POST /subscriptions",
        )

    async def pull_subscription(
        self,
        subscription_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[Event, ...]:
        _require_segment("subscription.id", subscription_id)
        params = {"limit": limit} if limit is not None else None
        return _model_tuple_response(
            await self._request(
                "POST",
                f"/subscriptions/{subscription_id}/pull",
                params=params,
            ),
            Event,
            "POST /subscriptions/{id}/pull",
        )

    async def get_subscription(self, subscription_id: str) -> Subscription:
        _require_segment("subscription.id", subscription_id)
        return _model_response(
            await self._request("GET", f"/subscriptions/{subscription_id}"),
            Subscription,
            "GET /subscriptions/{id}",
        )

    async def write_artifact(
        self,
        data: bytes | str,
        *,
        channel: str | None = None,
        media_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
        event_ids: tuple[str, ...] = (),
    ) -> Artifact:
        _require_optional_segment("artifact.channel", channel)
        event_id_list = _segment_list("artifact.event_ids", event_ids)
        payload = _artifact_payload(data)
        return _model_response(
            (
                await self._request(
                    "POST",
                    "/artifacts",
                    json={
                        "data": payload["data"],
                        "encoding": payload["encoding"],
                        "channel": channel,
                        "media_type": media_type,
                        "metadata": _copy_mapping("artifact.metadata", metadata),
                        "event_ids": event_id_list,
                    },
                )
            ),
            Artifact,
            "POST /artifacts",
        )

    async def read_artifact(self, artifact_id: str) -> bytes:
        _require_segment("artifact.id", artifact_id)
        return (await self._request("GET", f"/artifacts/{artifact_id}")).content

    async def get_artifact(self, artifact_id: str) -> Artifact:
        _require_segment("artifact.id", artifact_id)
        return _model_response(
            await self._request("GET", f"/artifacts/{artifact_id}/metadata"),
            Artifact,
            "GET /artifacts/{id}/metadata",
        )

    async def put_snapshot(
        self,
        name: str | Snapshot | Mapping[str, Any],
        value: Any = None,
        *,
        channel: str | None = None,
        timestamp: float | None = None,
        schema: str | None = None,
        source_event_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Snapshot:
        """Write a snapshot given as a model, a dict, or a name and value.

        ``put_snapshot(snapshot)`` takes a ``Snapshot`` or a dict of snapshot
        fields; ``put_snapshot(name, value, ...)`` stores ``value`` exactly as
        given, unless it is a ``Snapshot`` whose fields are used under that
        name. Field arguments that are not ``None`` override.
        """
        name, payload = _snapshot_payload(
            name,
            value,
            channel=channel,
            timestamp=timestamp,
            schema=schema,
            source_event_id=source_event_id,
            metadata=metadata,
        )
        _require_optional_segment("snapshot.channel", payload.get("channel"))
        _require_optional_segment(
            "snapshot.source_event_id",
            payload.get("source_event_id"),
        )
        if "metadata" in payload:
            payload["metadata"] = _mapping_payload(
                "snapshot.metadata",
                payload["metadata"],
            )
        return _model_response(
            (
                await self._request(
                    "PUT",
                    f"/snapshots/{name}",
                    json=payload,
                )
            ),
            Snapshot,
            "PUT /snapshots/{name}",
        )

    async def get_snapshot(self, name: str) -> Snapshot:
        _require_segment("snapshot.name", name)
        return _model_response(
            await self._request("GET", f"/snapshots/{name}"),
            Snapshot,
            "GET /snapshots/{name}",
        )

    async def _request(self, method: str, path: str, **kwargs: Any):
        import httpx

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            response = await client.request(method, path, **kwargs)
        _raise_for_error(response)
        return response


def _dump_channel(channel: Channel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(channel, Channel):
        return copy_boundary_value(channel.model_dump(mode="json", by_alias=True))
    return _mapping_payload("channel", channel)


def _dump_event(event: Event | dict[str, Any]) -> dict[str, Any]:
    if isinstance(event, Event):
        return copy_boundary_value(event.model_dump(mode="json", by_alias=True))
    return _mapping_payload("event", event)


def _artifact_payload(data: bytes | str) -> dict[str, str]:
    if isinstance(data, bytes):
        return {
            "data": base64.b64encode(data).decode("ascii"),
            "encoding": "base64",
        }
    if not isinstance(data, str):
        raise InvalidPayloadError("artifact data must be bytes or text.")
    return {"data": data, "encoding": "text"}


def _base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("base_url must be a non-empty absolute http(s) URL")
    value = base_url
    if any(ch.isspace() for ch in value):
        raise ValueError("base_url must not contain whitespace")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute http(s) URL")
    return value.rstrip("/")


def _require_segment(field_name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "%" in value
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\")
    ):
        raise InvalidPayloadError(f"{field_name} must be a non-empty path segment.")


def _require_token(field_name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "%" in value
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\")
    ):
        raise InvalidPayloadError(f"{field_name} must be a non-empty token.")


def _require_optional_token(field_name: str, value: Any) -> None:
    if value is not None:
        _require_token(field_name, value)


def _require_optional_segment(field_name: str, value: Any) -> None:
    if value is not None:
        _require_segment(field_name, value)


def _segment_list(field_name: str, values: Any) -> list[str]:
    if (
        values is None
        or isinstance(values, (str, bytes, bytearray))
        or isinstance(values, Mapping)
        or not isinstance(values, Iterable)
    ):
        raise InvalidPayloadError(f"{field_name} must be a list of path segments.")
    result: list[str] = []
    for value in values:
        _require_segment(field_name, value)
        result.append(value)
    return result


def _require_segments(field_name: str, values: Any) -> None:
    _segment_list(field_name, values)


def _dump_snapshot(snapshot: Snapshot | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(snapshot, Snapshot):
        return copy_boundary_value(
            snapshot.model_dump(mode="json", by_alias=True, exclude_none=True)
        )
    return {
        key: item
        for key, item in _mapping_payload("snapshot", snapshot).items()
        if item is not None
    }


def _snapshot_payload(
    snapshot: Any,
    value: Any,
    *,
    channel: str | None,
    timestamp: float | None,
    schema: str | None,
    source_event_id: str | None,
    metadata: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if isinstance(snapshot, (Snapshot, Mapping)):
        payload = _dump_snapshot(snapshot)
        name = payload.pop("name", None)
        _require_segment("snapshot.name", name)
        if value is not None:
            payload["value"] = copy_boundary_value(value)
    else:
        _require_segment("snapshot.name", snapshot)
        name = snapshot
        if isinstance(value, Snapshot):
            payload = _dump_snapshot(value)
            payload.pop("name", None)
        else:
            payload = {"value": copy_boundary_value(value)}
    if channel is not None:
        payload["channel"] = channel
    if timestamp is not None:
        payload["timestamp"] = timestamp
    if schema is not None:
        payload["schema"] = schema
    if source_event_id is not None:
        payload["source_event_id"] = source_event_id
    if metadata is not None:
        payload["metadata"] = _mapping_payload("snapshot.metadata", metadata)
    return name, payload


def _mapping_payload(field_name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidPayloadError(f"{field_name} must be an object.")
    return copy_boundary_value(value)


def _copy_mapping(field_name: str, value: Any) -> dict[str, Any]:
    return _mapping_payload(field_name, value) if value is not None else {}


def _model_response(response: Any, model_type: Any, endpoint: str) -> Any:
    data = _response_json(response, endpoint)
    try:
        return model_type.model_validate(data)
    except Exception as exc:
        raise RemoteStoreError(
            response.status_code,
            error_type="InvalidResponse",
            message=f"LLLM network {endpoint} response did not match the expected schema.",
            detail=data,
        ) from exc


def _model_tuple_response(response: Any, model_type: Any, endpoint: str) -> tuple[Any, ...]:
    data = _response_json(response, endpoint)
    if not isinstance(data, list):
        raise RemoteStoreError(
            response.status_code,
            error_type="InvalidResponse",
            message=f"LLLM network {endpoint} response did not match the expected schema.",
            detail=data,
        )
    try:
        return tuple(model_type.model_validate(item) for item in data)
    except Exception as exc:
        raise RemoteStoreError(
            response.status_code,
            error_type="InvalidResponse",
            message=f"LLLM network {endpoint} response did not match the expected schema.",
            detail=data,
        ) from exc


def _response_json(response: Any, endpoint: str) -> Any:
    try:
        return response.json()
    except Exception as exc:
        raise RemoteStoreError(
            response.status_code,
            error_type="InvalidResponse",
            message=f"LLLM network {endpoint} response was not valid JSON.",
            detail=response.text,
        ) from exc


def _raise_for_error(response: Any) -> None:
    if response.status_code < 400:
        return
    try:
        data = response.json()
    except Exception:
        data = response.text
    error = _error_detail(data)
    raise RemoteStoreError(
        response.status_code,
        error_type=_error_token_field(error, "type"),
        message=_error_text_field(error, "message"),
        detail=data,
    )


def _error_detail(data: Any) -> Any:
    if isinstance(data, dict):
        detail = data.get("detail", data)
        if isinstance(detail, dict) and "error" in detail:
            return detail["error"]
        if "error" in data:
            return data["error"]
    return data


def _detail_message(detail: Any) -> str | None:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        error = _error_detail(detail)
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    return None


def _error_text_field(error: Any, field_name: str) -> str | None:
    if not isinstance(error, dict):
        return None
    value = error.get(field_name)
    return value if isinstance(value, str) else None


def _error_token_field(error: Any, field_name: str) -> str | None:
    value = _error_text_field(error, field_name)
    if value is None:
        return None
    try:
        return _token_value(value, f"error.{field_name}")
    except ValueError:
        return None


def _optional_text_value(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    return value


def _optional_token_value(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _token_value(value, label)


def _token_value(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if (
        not value.strip()
        or value in {".", ".."}
        or "%" in value
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\")
    ):
        raise ValueError(f"{label} must be a non-empty token.")
    return value
