"""Proxies: path values stay inside the documented endpoint, any placeholder
name routes, and the proxy project template answers ``CALL_API`` calls."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from lllm import load_package
from lllm.create import create_project
from lllm.runtimes.native import Runtime
from lllm.runtimes.native.proxies import BaseProxy, ProxyManager

ROOT = Path(__file__).resolve().parents[1]


class RecordingProxy(BaseProxy):
    """An HTTP-style proxy whose transport is recorded instead of sent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://api.example.test/v1"
        self.api_key_name = "apikey"
        self.api_key = "secret"
        self.sent = []

    def _call_api(self, url, params, endpoint_info, headers):
        self.sent.append((url, dict(params)))
        return {"url": url}

    @BaseProxy.endpoint(
        category="Health",
        endpoint="ping",
        description="Ping.",
        params={},
        response=[],
    )
    def ping(self, params):
        return params

    @BaseProxy.endpoint(
        category="Companies",
        endpoint="companies/{ticker}/kpi",
        description="Company KPIs.",
        params={"ticker*": (str, "AAPL")},
        response=[],
    )
    def company_kpi(self, params):
        return params

    @BaseProxy.endpoint(
        category="Series",
        endpoint="series/{series-id}/observations",
        description="Series observations.",
        params={"series-id*": (str, "GDP")},
        response=[],
    )
    def observations(self, params):
        return params


def manager():
    proxies = ProxyManager(runtime=Runtime())
    proxies.register("demo", RecordingProxy)
    return proxies


def test_placeholder_values_are_encoded_as_one_path_segment():
    proxies = manager()

    proxies("demo/companies/{ticker}/kpi", {"ticker": "../../admin/users?x="})
    proxies("demo/companies/{ticker}/kpi", {"ticker": "BRK/A"})

    assert proxies.proxies["demo"].sent == [
        (
            "https://api.example.test/v1/companies/..%2F..%2Fadmin%2Fusers%3Fx%3D/kpi",
            {"apikey": "secret"},
        ),
        ("https://api.example.test/v1/companies/BRK%2FA/kpi", {"apikey": "secret"}),
    ]


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("demo/companies/{ticker}/kpi", {}),
        ("demo/companies/{ticker}/kpi", {"ticker": None}),
        ("demo/companies/{ticker}/kpi", {"ticker": ""}),
        ("demo/companies/{ticker}/kpi", {"ticker": ".."}),
        ("demo/companies/../kpi", {}),
        ("demo/companies/./kpi", {}),
    ],
)
def test_missing_and_dot_placeholder_values_are_rejected_before_sending(
    path, params
):
    proxies = manager()

    with pytest.raises(ValueError, match="needs a value for 'ticker'"):
        proxies(path, params)

    assert proxies.proxies["demo"].sent == []


def test_placeholder_names_need_not_be_regex_group_names():
    proxies = manager()

    proxies("demo/ping", {})
    proxies("demo/series/GDP/observations", {})
    proxies("demo/series/{series-id}/observations", {"series-id": "CPI"})

    assert proxies.proxies["demo"].sent == [
        ("https://api.example.test/v1/ping", {"apikey": "secret"}),
        ("https://api.example.test/v1/series/GDP/observations", {"apikey": "secret"}),
        ("https://api.example.test/v1/series/CPI/observations", {"apikey": "secret"}),
    ]


def test_proxy_template_endpoints_answer_calls_through_the_proxy_manager(tmp_path):
    project = create_project("proxy", "proxy-app", directory=tmp_path).path
    runtime = Runtime()
    load_package(project / "lllm.toml", runtime=runtime)
    proxies = ProxyManager(activate_proxies=["sample"], runtime=runtime)

    assert proxies("sample/accounts", {"limit": 2}) == [
        {"id": "acct_001", "name": "Northwind", "tier": "enterprise"},
        {"id": "acct_002", "name": "Contoso", "tier": "startup"},
    ]
    assert proxies("sample/events", {"account_id": "acct_002", "limit": 1}) == [
        {"account_id": "acct_002", "event": "login", "count": 48},
    ]


def test_proxy_template_catalog_lists_the_sample_endpoints(tmp_path):
    project = create_project("proxy", "proxy-app", directory=tmp_path).path
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    completed = subprocess.run(
        [sys.executable, "main.py", "--catalog"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "**/accounts**" in completed.stdout
    assert "**/events**" in completed.stdout
