"""Canned model replies for docs/configuration.md (see docs_model.py)."""

from __future__ import annotations

from docs_model import asks, rule, system_has


@rule(asks("write release notes for:", "login timeout"))
def _release_draft(request):
    return "Startup is now faster, and the login timeout bug is fixed."


@rule(asks("tighten these release notes"))
def _release_edit(request):
    return "Faster startup, and login no longer times out."


@rule(
    lambda r: system_has("energy markets")(r)
    and asks("why did oil prices fall")(r)
)
def _oil(request):
    return "Oil fell on rising inventories and weaker demand forecasts."


@rule(system_has("translate into french"))
def _french(request):
    return "Bonjour" if "good morning" in request.text.lower() else "Merci"
