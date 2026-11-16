"""Canned model replies for docs/proxies.md (see docs_model.py).

The trip planner reads the weather proxy's docs, writes one program that
calls the endpoint for both cities, and answers from what it printed.
"""

from __future__ import annotations

import ast

from docs_model import rule, tool_result
from lllm.testing import ToolCall


def _user_asked(request, phrase: str) -> bool:
    return any(
        phrase.lower() in str(message.get("content") or "").lower()
        for message in request.messages
        if message.get("role") == "user"
    )


def _last_tool(request) -> str | None:
    last = request.last_message
    return last.get("name") if last.get("role") == "tool" else None


def _planner(request) -> bool:
    return "run_python" in request.tools and _user_asked(request, "which will be drier")


@rule(lambda r: _planner(r) and not r.tool_results())
def _planner_reads_docs(request):
    return ToolCall("query_api_doc", {"proxy_name": "weather"})


@rule(lambda r: _planner(r) and _last_tool(r) == "query_api_doc")
def _planner_writes_code(request):
    code = (
        'for city in ["Tokyo", "Paris"]:\n'
        '    print(city, CALL_API(f"weather/forecast/{city}"))'
    )
    return ToolCall("run_python", {"code": code})


@rule(lambda r: _planner(r) and _last_tool(r) == "run_python")
def _planner_answers(request):
    forecasts = {}
    for line in tool_result(request).splitlines():
        city, _, data = line.partition(" ")
        forecasts[city] = ast.literal_eval(data)
    paris, tokyo = forecasts["Paris"], forecasts["Tokyo"]
    sky = {"sun": "sunny", "rain": "rain"}
    return (
        f"Paris will be drier: {sky[paris['sky']]}, {paris['low']} to "
        f"{paris['high']}°C. Tokyo expects {sky[tokyo['sky']]}."
    )
