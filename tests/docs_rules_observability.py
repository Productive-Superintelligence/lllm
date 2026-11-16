"""Canned model replies for docs/observability.md (see docs_model.py).

The triage agent looks up the customer's plan, answers once in prose (which
fails the ``Ticket`` format and is repaired), then answers in JSON, so its
session shows a tool round, a repair, and three model calls.
"""

from __future__ import annotations

import json

from docs_model import asks, rule
from lllm.testing import ToolCall


def _user_asked(request, phrase: str) -> bool:
    return any(
        phrase.lower() in str(message.get("content") or "").lower()
        for message in request.messages
        if message.get("role") == "user"
    )


def _triage(request) -> bool:
    return "support_plan" in request.tools and _user_asked(request, "dashboard is down")


@rule(lambda r: _triage(r) and r.text.startswith("Error:"))
def _triage_repaired(request):
    return json.dumps(
        {"priority": "high", "reason": "Enterprise customer with an outage."}
    )


@rule(lambda r: _triage(r) and not r.tool_results())
def _triage_looks_up_plan(request):
    return ToolCall("support_plan", {"customer": "ACME"})


@rule(lambda r: _triage(r) and bool(r.tool_results()))
def _triage_answers_in_prose(request):
    return "High priority: an enterprise customer has an outage."  # not JSON


FRENCH = {"Good morning": "Bonjour", "Thank you": "Merci"}


def _after_colon(request) -> str:
    return request.text.split(":", 1)[-1].strip()


@rule(lambda r: asks("translate to french:")(r) and _after_colon(r) in FRENCH)
def _translate(request):
    return FRENCH[_after_colon(request)]


SUMMARIES = {
    "Sessions record every agent call and its cost.": (
        "Sessions record calls and cost."
    ),
    "Observers save sessions wherever you keep runs.": "Observers save every run.",
}


@rule(lambda r: asks("summarize in five words:")(r) and _after_colon(r) in SUMMARIES)
def _five_words(request):
    return SUMMARIES[_after_colon(request)]
