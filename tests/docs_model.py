"""Canned model replies for the docs examples (see test_docs_examples.py).

``answer`` plays the model for every example in docs/. Each rule matches
what an example asks (usually a phrase from the latest user message, or the
system prompt) and returns the reply a real model would plausibly give, so
the outputs shown in the docs are realistic and checked on every run.
Keep replies short and deterministic.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from lllm.testing import ModelRequest, Reply, ToolCall

Rule = tuple[Callable[[ModelRequest], bool], Callable[[ModelRequest], Reply]]
RULES: list[Rule] = []


def rule(match: Callable[[ModelRequest], bool]):
    def register(reply: Callable[[ModelRequest], Reply]):
        RULES.append((match, reply))
        return reply

    return register


def asks(*phrases: str) -> Callable[[ModelRequest], bool]:
    """Match when the latest message contains every phrase (case-insensitive)."""

    return lambda request: all(p.lower() in request.text.lower() for p in phrases)


def system_has(*phrases: str) -> Callable[[ModelRequest], bool]:
    return lambda request: all(p.lower() in request.system.lower() for p in phrases)


def tool_result(request: ModelRequest) -> str:
    """The raw value of the latest tool result (without the runtime's framing)."""

    text = request.tool_results()[-1]
    match = re.search(r"---\n(.*)\n---", text, re.S)
    return match.group(1).strip() if match else text


def answer(request: ModelRequest) -> Reply:
    for match, reply in RULES:
        if match(request):
            return reply(request)
    raise AssertionError(
        "docs_model has no reply for this request; add a rule in tests/docs_model.py.\n"
        f"system: {request.system[:200]!r}\nlast message: {request.text[:400]!r}"
    )


# -- Introduction -------------------------------------------------------------


@rule(asks("hello world", "come from"))
def _hello(request):
    return "It comes from Brian Kernighan's 1972 tutorial for the B language."


@rule(lambda r: "search" in r.tools and r.text.lower().startswith("collect facts about"))
def _research_start(request):
    topic = request.text.split("about", 1)[1].strip(" .")
    return ToolCall("search", {"query": topic})


@rule(lambda r: "search" in r.tools and bool(r.tool_results()))
def _research_notes(request):
    return f"Notes: {tool_result(request)}"


@rule(asks("three-bullet brief about", "kv caches"))
def _brief(request):
    return json.dumps(
        {
            "title": "KV caches in one minute",
            "bullets": [
                "A KV cache stores attention keys and values from earlier tokens.",
                "Reusing it makes each new token cost one step, not a full pass.",
                "Shared prompt prefixes can reuse cached entries across requests.",
            ],
        }
    )


# -- How LLLM works -------------------------------------------------------------


@rule(lambda r: "to_fahrenheit" in r.tools and not r.tool_results())
def _units_call(request):
    celsius = float(re.search(r"([-\d.]+)°C", request.text).group(1))
    return ToolCall("to_fahrenheit", {"celsius": celsius if celsius % 1 else int(celsius)})


@rule(lambda r: "to_fahrenheit" in r.tools and bool(r.tool_results()))
def _units_answer(request):
    return "21°C is 69.8°F."


# -- Agents ------------------------------------------------------------------------


def _said(request, pattern):
    for message in request.messages:
        if message.get("role") == "user":
            found = re.search(pattern, str(message.get("content") or ""))
            if found:
                return found
    return None


@rule(asks("my name is"))
def _greet(request):
    return f"Nice to meet you, {_said(request, r'My name is (\w+)').group(1)}!"


@rule(asks("what is my name"))
def _recall_name(request):
    found = _said(request, r"My name is (\w+)")
    return f"Your name is {found.group(1)}." if found else "You have not told me yet."


@rule(asks("capital of france"))
def _paris(request):
    return "The capital of France is Paris."


@rule(asks("one-line project update"))
def _update(request):
    return "The docs now run every example and check its output."


@rule(asks("plan a launch checklist"))
def _plan(request):
    return "1. Freeze the release branch. 2. Run the rollback test. 3. Announce the launch."


@rule(asks("biggest risk in this plan"))
def _risk(request):
    return "The biggest risk is skipping the rollback test."


@rule(lambda r: r.text.startswith("Error:") and "please fix" in r.text.lower() and bool(_said(r, r"capital of Japan")))
def _tokyo_repaired(request):
    return json.dumps({"name": "Tokyo", "country": "Japan"})


@rule(asks("name the capital of japan"))
def _tokyo_first(request):
    return "Tokyo"  # not JSON: the agent repairs it


@rule(asks("name for a caching library"))
def _name(request):
    return "Stash"


@rule(asks("what is 2 + 2"))
def _four(request):
    return "4"


# -- Prompts --------------------------------------------------------------------------


@rule(asks("classify this ticket", "charged twice"))
def _ticket(request):
    return "<label>billing</label>"


@rule(asks("extract the invoice fields", "ACME"))
def _invoice(request):
    return json.dumps({"vendor": "ACME Corp", "total": 1250.0, "currency": "EUR"})


# -- Tools ----------------------------------------------------------------------------


@rule(lambda r: "lookup_order" in r.tools and not r.tool_results())
def _order_call(request):
    return ToolCall("lookup_order", {"order_id": re.search(r"[A-Z]-\d+", request.text).group(0)})


@rule(lambda r: "lookup_order" in r.tools and bool(r.tool_results()))
def _order_answer(request):
    return "Your order A-1001 has shipped and should arrive on Thursday."


@rule(lambda r: "fact_check" in r.tools and not r.tool_results())
def _fact_check_call(request):
    return ToolCall("fact_check", {"task": "Paris is the capital of France"})


@rule(lambda r: "fact_check" in r.tools and bool(r.tool_results()))
def _fact_check_answer(request):
    return f"Paris is the capital of France (checked: {tool_result(request)})."
