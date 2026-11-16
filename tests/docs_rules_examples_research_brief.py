"""Canned model replies for docs/examples/research-brief.md (see docs_model.py)."""

from __future__ import annotations

import json
import re

from docs_model import asks, rule, tool_result
from lllm.testing import ToolCall

QUESTIONS = [
    "What made deploys fail last quarter?",
    "What fixes were rolled out?",
    "Did the failure rate improve after the fixes?",
]
KEYWORDS = {
    QUESTIONS[0]: ["deploys failed"],
    QUESTIONS[1]: ["fix", "deploy"],
    QUESTIONS[2]: ["failure rate"],
}
ANSWERS = {
    QUESTIONS[0]: "Migrations outran the five-minute deploy timeout, and two "
    "pipelines raced for the same deploy lock.",
    QUESTIONS[1]: "Long migrations now run before the deploy starts, and "
    "deploys are queued per service.",
    QUESTIONS[2]: "Yes. The deploy failure rate fell from 9% in Q1 to 2% in Q2.",
}


def _question(request) -> str:
    for message in request.messages:
        content = str(message.get("content") or "")
        if message.get("role") == "user" and "in the incident reports:" in content:
            return content.split("in the incident reports:", 1)[1].splitlines()[0].strip()
    return ""


@rule(asks("break this question into three research questions", "deploys fail"))
def _plan(request):
    return json.dumps({"questions": QUESTIONS})


@rule(lambda r: "search_reports" in r.tools and not r.tool_results())
def _search(request):
    return ToolCall("search_reports", {"keywords": KEYWORDS[_question(request)]})


@rule(lambda r: "search_reports" in r.tools and bool(r.tool_results()))
def _finding(request):
    question = _question(request)
    sources = re.findall(r"^(INC-\d+):", tool_result(request), re.M)
    return json.dumps({"question": question, "answer": ANSWERS[question], "sources": sources})


@rule(asks("write a brief that answers", "deploys fail"))
def _brief(request):
    sources = list(dict.fromkeys(re.findall(r"INC-\d+", request.text)))
    return json.dumps(
        {
            "title": "Why deploys failed last quarter",
            "points": [
                "Migrations outran the five-minute deploy timeout (INC-101).",
                "Two pipelines raced for the same deploy lock (INC-103).",
                "Migrations now run before deploys; deploys queue per service.",
                "Failures fell from 9% in Q1 to 2% in Q2 (INC-104).",
            ],
            "sources": sources,
        }
    )
