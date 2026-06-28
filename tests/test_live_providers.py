"""Opt-in provider key smoke checks.

These tests only list models. They do not send prompts or request generations.
Run them with ``LLLM_LIVE_PROVIDER_TESTS=1`` when checking local credentials.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("LLLM_LIVE_PROVIDER_TESTS") != "1",
        reason="set LLLM_LIVE_PROVIDER_TESTS=1 to run live provider checks",
    ),
]


def test_openai_key_lists_models():
    result = _list_models(
        "https://api.openai.com/v1/models",
        {"Authorization": f"Bearer {_required_env('OPENAI_API_KEY')}"},
    )

    assert result.ok
    assert result.model_count > 0


def test_anthropic_key_lists_models():
    result = _list_models(
        "https://api.anthropic.com/v1/models",
        {
            "x-api-key": _required_env("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01",
        },
    )

    assert result.ok
    assert result.model_count > 0


@pytest.mark.xfail(
    reason=(
        "Together may return edge-level 403 error code 1010 from some networks "
        "before API-key validation."
    ),
    strict=False,
)
def test_together_key_lists_models():
    result = _list_models(
        "https://api.together.ai/v1/models",
        {"Authorization": f"Bearer {_required_env('TOGETHER_API_KEY')}"},
    )

    assert result.ok
    assert result.model_count > 0


class ProviderResult:
    def __init__(self, *, ok: bool, status: int, model_count: int = 0) -> None:
        self.ok = ok
        self.status = status
        self.model_count = model_count


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set")
    return value


def _list_models(url: str, headers: dict[str, str]) -> ProviderResult:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return ProviderResult(
                ok=response.status == 200,
                status=response.status,
                model_count=len(_model_items(payload)),
            )
    except urllib.error.HTTPError as exc:
        return ProviderResult(ok=False, status=exc.code)


def _model_items(payload: object) -> list[object]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
        models = payload.get("models")
        if isinstance(models, list):
            return models
    if isinstance(payload, list):
        return payload
    return []
