"""Exercise the example through the real Pydantic AI MCP transport and LLLM wrapper."""

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

pytest.importorskip("pydantic_ai.mcp")
from pydantic_ai.mcp import MCPToolset

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/parallel_search/agent.py"
spec = importlib.util.spec_from_file_location("parallel_search_example", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_mcp_tools_through_tactic(monkeypatch):
    requests = []
    schemas = [
        {"name": "web_search", "description": "Search", "inputSchema": {
            "type": "object", "properties": {
                "objective": {"type": "string"},
                "search_queries": {"type": "array", "items": {"type": "string"}},
                "session_id": {"type": "string"},
            }, "required": ["objective", "search_queries"],
        }},
        {"name": "web_fetch", "description": "Fetch", "inputSchema": {
            "type": "object", "properties": {
                "urls": {"type": "array", "items": {"type": "string"}},
                "objective": {"type": "string"},
                "session_id": {"type": "string"},
            }, "required": ["urls"],
        }},
    ]

    async def send(client, request, **kwargs):
        assert str(request.url) == example.MCP_URL
        assert request.headers["user-agent"] == example.USER_AGENT
        assert "authorization" not in request.headers
        if request.method != "POST":
            return httpx.Response(405, request=request)
        payload = json.loads(request.content)
        requests.append(payload)
        method = payload["method"]
        if "id" not in payload:
            return httpx.Response(202, request=request)
        if method == "initialize":
            result = {
                "protocolVersion": payload["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-search", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": schemas}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": json.dumps({
                "url": "https://github.com/Productive-Superintelligence/lllm",
                "excerpt": "LLLM exposes reusable typed tactics.",
            })}], "isError": False}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": payload["id"], "result": result,
        }, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)

    async def run():
        tactic = example.build_tactic(example.smoke_model())
        assert any(isinstance(toolset, MCPToolset) for toolset in tactic.agent.toolsets)
        async with tactic.agent:
            return await tactic.arun("What is LLLM?")

    output = json.loads(asyncio.run(run()))
    assert set(output) == {"web_search", "web_fetch"}
    assert "reusable typed tactics" in json.dumps(output)
    assert any(p["method"] == "tools/list" for p in requests)
    calls = [p["params"] for p in requests if p["method"] == "tools/call"]
    assert [p["name"] for p in calls] == ["web_search", "web_fetch"]
    assert calls[0]["arguments"]["session_id"] == calls[1]["arguments"]["session_id"]
    assert calls[0]["arguments"]["search_queries"]
    assert calls[1]["arguments"]["urls"] == [
        "https://github.com/Productive-Superintelligence/lllm"
    ]
