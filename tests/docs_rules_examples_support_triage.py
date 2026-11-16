"""Canned model replies for docs/examples/support-triage.md (see docs_model.py)."""

from __future__ import annotations

import re

from docs_model import asks, rule
from lllm.testing import ToolCall

# keyword in the ticket -> (category, urgency)
TRIAGE = {
    "charges": ("billing", "low"),
    "export": ("technical", "low"),
    "refund": ("billing", "low"),
    "locked out": ("account", "high"),
}


def _ticket(request) -> str:
    for message in request.messages:
        content = str(message.get("content") or "")
        if message.get("role") == "user" and "ticket from" in content:
            return content
    return ""


def _last_tool(request) -> str | None:
    last = request.last_message
    return last.get("name") if last.get("role") == "tool" else None


@rule(asks("sort this support ticket"))
def _triage(request):
    ticket = request.text.split("Ticket:", 1)[1].splitlines()[0].lower()
    category, urgency = next(v for k, v in TRIAGE.items() if k in ticket)
    return f"<category>{category}</category>\n<urgency>{urgency}</urgency>"


@rule(lambda r: "issue_refund" in r.tools and not r.tool_results())
def _billing_invoices(request):
    customer = re.search(r"ticket from (\w+):", _ticket(request)).group(1)
    return ToolCall("get_invoices", {"customer": customer})


@rule(lambda r: "issue_refund" in r.tools and _last_tool(r) == "get_invoices")
def _billing_refund(request):
    ticket = _ticket(request)
    customer = re.search(r"ticket from (\w+):", ticket).group(1)
    amount = float(re.search(r"\$(\d+)", ticket).group(1))
    return ToolCall("issue_refund", {"customer": customer, "amount": amount})


@rule(lambda r: "issue_refund" in r.tools and _last_tool(r) == "issue_refund")
def _billing_reply(request):
    if request.text.startswith("Error:"):
        return (
            "<reply>A teammate will approve the refund.</reply>\n"
            "<ESCALATE>"
        )
    return "<reply>Refunded the duplicate $30 charge.</reply>"


@rule(lambda r: "search_known_issues" in r.tools and not r.tool_results())
def _technical_search(request):
    return ToolCall("search_known_issues", {"keyword": "export"})


@rule(lambda r: "search_known_issues" in r.tools and bool(r.tool_results()))
def _technical_reply(request):
    return "<reply>Known issue KI-7; a fix ships Friday.</reply>"
