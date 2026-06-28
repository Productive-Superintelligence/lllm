"""HTTP client for remote tactic services."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..protocol import CallContext, Tactic, TacticServiceError


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
        self.error_type = error_type
        self.tactic = tactic
        self.endpoint = endpoint
        self.request_id = request_id
        self.detail = detail
        self.message = message or _detail_message(detail)
        text = f"Remote tactic returned HTTP {status_code}"
        if error_type:
            text += f" ({error_type})"
        if self.message:
            text += f": {self.message}"
        super().__init__(text)


class RemoteTactic(Tactic[Any, Any]):
    """Call a tactic through its HTTP `/run` service endpoint."""

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
        self.url = _run_url(url)
        self.timeout = timeout
        self.transport = transport
        self.async_transport = async_transport
        self.input_type = input_type
        self.output_type = output_type
        super().__init__(
            name=name or _name_from_url(url),
            service_ref=url,
            metadata={"url": self.url, **dict(metadata or {})},
        )

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
        value = input_value.model_dump(mode="json")
    else:
        value = input_value
    return {
        "input": value,
        "context": (context or CallContext()).model_dump(mode="json"),
    }


def _response_output(response: Any) -> Any:
    if response.status_code >= 400:
        raise _remote_error(response)
    data = response.json()
    if isinstance(data, dict) and "output" in data:
        return data["output"]
    return data


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
        error_type=error.get("type") if isinstance(error, dict) else None,
        message=error.get("message") if isinstance(error, dict) else None,
        tactic=error.get("tactic") if isinstance(error, dict) else None,
        endpoint=error.get("endpoint") if isinstance(error, dict) else None,
        request_id=error.get("request_id") if isinstance(error, dict) else None,
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


def _run_url(url: str) -> str:
    value = url.rstrip("/")
    if value.endswith("/run"):
        return value
    return f"{value}/run"


def _name_from_url(url: str) -> str:
    parts = [part for part in url.rstrip("/").split("/") if part]
    if not parts:
        return "remote"
    if parts[-1] == "run" and len(parts) > 1:
        return parts[-2]
    return parts[-1]
