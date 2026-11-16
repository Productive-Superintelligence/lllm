"""Canned model replies for docs/runtimes.md (see docs_model.py)."""

from __future__ import annotations

from docs_model import asks, rule


@rule(asks("write a headline for", "error rate crosses"))
def _headline(request):
    return "Deploys Pause When Errors Top 2%"
