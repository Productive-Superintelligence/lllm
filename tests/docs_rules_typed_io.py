"""Canned model replies for docs/typed-io.md (see docs_model.py)."""

from __future__ import annotations

import json

from docs_model import asks, rule


def _said(request, phrase: str) -> bool:
    return any(
        message.get("role") == "user"
        and phrase in str(message.get("content") or "").lower()
        for message in request.messages
    )


# -- Opening example ----------------------------------------------------------


@rule(asks("route this ticket.", "charged twice"))
def _route_charge(request):
    return json.dumps({"team": "billing", "priority": 2})


# -- Repairing a reply that does not fit --------------------------------------


@rule(
    lambda r: r.text.startswith(
        "Error: The reply does not match Triage: team: Input should be"
    )
    and _said(r, "wrong company name")
)
def _triage_repaired(request):
    return json.dumps({"team": "billing", "priority": 3})


@rule(asks("route this ticket to billing, bugs, or accounts", "wrong company name"))
def _triage_first(request):
    return json.dumps({"team": "invoicing", "priority": 3})  # not a team
