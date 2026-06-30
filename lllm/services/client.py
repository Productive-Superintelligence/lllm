"""HTTP client for remote tactic services."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel

from ..protocol import (
    CallContext,
    Tactic,
    TacticEvent,
    TacticInfo,
    TacticServiceError,
)
from ..protocol._validation import copy_boundary_value, optional_mapping_value, token_value


class RemoteTacticError(TacticServiceError):
    """Raised when a remote tactic service returns a structured error."""

    def __init__(
        self,
        status_code: int,
        *,
        error_type: str | None = None,
        message: str | None = None,
        tactic: str | None = None,
        endpoint: str | None = None,
        request_id: str | None = None,
        detail: Any = None,
    ) -> None:
        self.status_code = status_code
        self.error_type = _optional_token_value(error_type, "error_type")
        self.tactic = _optional_text_value(tactic, "tactic")
        self.endpoint = _optional_text_value(endpoint, "endpoint")
        self.request_id = _optional_token_value(request_id, "request_id")
        self.detail = detail
        self.message = _optional_text_value(message, "message") or _detail_message(detail)
        text = f"Remote tactic returned HTTP {status_code}"
        if self.error_type:
            text += f" ({self.error_type})"
        if self.message:
            text += f": {self.message}"
        super().__init__(text)


class RemoteTactic(Tactic[Any, Any]):
    """Call a tactic through its HTTP service endpoints."""

    runtime_kind = "http"

    def __init__(
        self,
        url: str,
        *,
        name: str | None = None,
        input_type: Any = None,
        output_type: Any = None,
        timeout: float | None = 30.0,
        transport: Any = None,
        async_transport: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        url = _service_url(url)
        self.url = _run_url(url)
        self.stream_url = _stream_url(url)
        self.info_url = _info_url(url)
        self.timeout = timeout
        self.transport = transport
        self.async_transport = async_transport
        self.input_type = input_type
        self.output_type = output_type
        metadata_value = _metadata_mapping(metadata)
        metadata_value["url"] = self.url
        super().__init__(
            name=name if name is not None else _name_from_url(url),
            service_ref=url,
            metadata=metadata_value,
        )

    def fetch_info(self, **kwargs: Any) -> TacticInfo:
        """Fetch the service-advertised tactic contract."""

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Install httpx or lllm[client] to inspect remote tactics."
            ) from exc

        with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
            response = client.get(self.info_url, **kwargs)
        return _response_info(response)

    async def afetch_info(self, **kwargs: Any) -> TacticInfo:
        """Fetch the service-advertised tactic contract asynchronously."""

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Install httpx or lllm[client] to inspect remote tactics."
            ) from exc

        async with httpx.AsyncClient(
            transport=self.async_transport,
            timeout=self.timeout,
        ) as client:
            response = await client.get(self.info_url, **kwargs)
        return _response_info(response)

    def _run(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install httpx or lllm[client] to call remote tactics.") from exc

        envelope = _request_envelope(input_value, context)
        with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
            response = client.post(self.url, json=envelope, **kwargs)
        return _response_output(response)

    def stream(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install httpx or lllm[client] to stream remote tactics.") from exc

        envelope = _request_envelope(input_value, context)
        with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
            with client.stream("POST", self.stream_url, json=envelope, **kwargs) as response:
                if response.status_code >= 400:
                    response.read()
                    raise _remote_error(response)
                for event in _iter_sse_events(response.iter_lines()):
                    yield event.data

    async def astream(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async for event in self.aevents(input_value, context=context, **kwargs):
            yield event.data

    async def aevents(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[TacticEvent]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install httpx or lllm[client] to stream remote tactics.") from exc

        envelope = _request_envelope(input_value, context)
        async with httpx.AsyncClient(
            transport=self.async_transport,
            timeout=self.timeout,
        ) as client:
            async with client.stream(
                "POST",
                self.stream_url,
                json=envelope,
                **kwargs,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise _remote_error(response)
                async for event in _aiter_sse_events(response.aiter_lines()):
                    yield event

    async def _arun(
        self,
        input_value: Any,
        *,
        context: CallContext | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install httpx or lllm[client] to call remote tactics.") from exc

        envelope = _request_envelope(input_value, context)
        async with httpx.AsyncClient(
            transport=self.async_transport,
            timeout=self.timeout,
        ) as client:
            response = await client.post(self.url, json=envelope, **kwargs)
        return _response_output(response)


def _request_envelope(input_value: Any, context: CallContext | None) -> dict[str, Any]:
    if isinstance(input_value, BaseModel):
        value = copy_boundary_value(input_value.model_dump(mode="json"))
    else:
        value = copy_boundary_value(input_value)
    context_value = _call_context(context)
    return {
        "input": value,
        "context": copy_boundary_value(context_value.model_dump(mode="json")),
    }


def _call_context(value: Any) -> CallContext:
    if value is None:
        return CallContext()
    if isinstance(value, CallContext):
        return value
    raise TypeError("context must be a CallContext.")


def _metadata_mapping(value: Any) -> dict[str, Any]:
    return optional_mapping_value("metadata", value)


def _response_output(response: Any) -> Any:
    if response.status_code >= 400:
        raise _remote_error(response)
    data = _response_json(response, endpoint="run")
    if isinstance(data, dict) and "output" in data:
        return data["output"]
    return data


def _response_info(response: Any) -> TacticInfo:
    if response.status_code >= 400:
        raise _remote_error(response)
    data = _response_json(response, endpoint="info")
    try:
        return TacticInfo.model_validate(data)
    except Exception as exc:
        raise RemoteTacticError(
            response.status_code,
            error_type="InvalidResponse",
            message="Remote tactic info response did not match the TacticInfo schema.",
            endpoint="info",
            detail=data,
        ) from exc


def _response_json(response: Any, *, endpoint: str) -> Any:
    try:
        return response.json()
    except Exception as exc:
        raise RemoteTacticError(
            response.status_code,
            error_type="InvalidResponse",
            message=f"Remote tactic {endpoint} response was not valid JSON.",
            endpoint=endpoint,
            detail=response.text,
        ) from exc


def _remote_error(response: Any) -> RemoteTacticError:
    try:
        data = response.json()
    except Exception:
        return RemoteTacticError(
            response.status_code,
            message=response.text,
            detail=response.text,
        )
    error = _error_detail(data)
    return RemoteTacticError(
        response.status_code,
        error_type=_error_token_field(error, "type"),
        message=_error_text_field(error, "message"),
        tactic=_error_text_field(error, "tactic"),
        endpoint=_error_text_field(error, "endpoint"),
        request_id=_error_token_field(error, "request_id"),
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
        return token_value(value, f"error.{field_name}")
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
    return token_value(value, label)


def _iter_sse_events(lines: Iterable[Any]) -> Iterator[TacticEvent]:
    buffer: list[str] = []
    for line in lines:
        try:
            text = _sse_line(line)
        except (TypeError, UnicodeDecodeError) as exc:
            if buffer:
                yield _sse_event(buffer)
                buffer = []
            yield _invalid_sse_line_event(line, exc)
            continue
        if text == "":
            if buffer:
                yield _sse_event(buffer)
                buffer = []
            continue
        data = _sse_data(text)
        if data is not None:
            buffer.append(data)
    if buffer:
        yield _sse_event(buffer)


async def _aiter_sse_events(lines: AsyncIterator[Any]) -> AsyncIterator[TacticEvent]:
    buffer: list[str] = []
    async for line in lines:
        try:
            text = _sse_line(line)
        except (TypeError, UnicodeDecodeError) as exc:
            if buffer:
                yield _sse_event(buffer)
                buffer = []
            yield _invalid_sse_line_event(line, exc)
            continue
        if text == "":
            if buffer:
                yield _sse_event(buffer)
                buffer = []
            continue
        data = _sse_data(text)
        if data is not None:
            buffer.append(data)
    if buffer:
        yield _sse_event(buffer)


def _sse_event(data_lines: list[str]) -> TacticEvent:
    payload = "\n".join(data_lines)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        value = payload
    if isinstance(value, dict):
        if _is_tactic_event_payload(value):
            try:
                return TacticEvent.model_validate(value)
            except Exception as exc:
                return TacticEvent.error(
                    "Invalid SSE event envelope.",
                    payload=value,
                    errors=_validation_errors(exc),
                )
        return TacticEvent(data=value)
    return TacticEvent(data=value)


def _is_tactic_event_payload(value: dict[str, Any]) -> bool:
    return "kind" in value and "data" in value


def _validation_errors(exc: Exception) -> Any:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        return errors()
    return str(exc)


def _sse_data(text: str) -> str | None:
    if not text.startswith("data:"):
        return None
    value = text[5:]
    if value.startswith(" "):
        value = value[1:]
    return value


def _sse_line(line: Any) -> str:
    if isinstance(line, str):
        return line
    if isinstance(line, bytes):
        return line.decode("utf-8")
    raise TypeError("SSE stream lines must be text or bytes.")


def _invalid_sse_line_event(line: Any, exc: Exception) -> TacticEvent:
    return TacticEvent.error(
        "Invalid SSE stream line.",
        line_type=type(line).__name__,
        errors=_validation_errors(exc),
    )


def _run_url(url: str) -> str:
    return _endpoint_url(url, "run")


def _stream_url(url: str) -> str:
    return _endpoint_url(url, "stream")


def _info_url(url: str) -> str:
    return _endpoint_url(url, "info")


def _endpoint_url(url: str, endpoint: str) -> str:
    value = url.rstrip("/")
    if (
        value.endswith("/run")
        or value.endswith("/stream")
        or value.endswith("/info")
    ):
        return f"{value.rsplit('/', 1)[0]}/{endpoint}"
    if value.endswith(f"/{endpoint}"):
        return value
    return f"{value}/{endpoint}"


def _service_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        raise ValueError("url must be a non-empty absolute http(s) URL")
    value = url
    if any(ch.isspace() for ch in value):
        raise ValueError("url must not contain whitespace")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an absolute http(s) URL")
    return value


def _name_from_url(url: str) -> str:
    parts = [part for part in url.rstrip("/").split("/") if part]
    if not parts:
        return "remote"
    if parts[-1] in {"run", "stream", "info"} and len(parts) > 1:
        return parts[-2]
    return parts[-1]
