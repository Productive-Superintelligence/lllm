"""Canned model replies for docs/packages.md (see docs_model.py)."""

from __future__ import annotations

from docs_model import asks, rule


@rule(asks("one-line greeting for new customers"))
def _greeting(request):
    return "Welcome aboard! We are glad you are here."
