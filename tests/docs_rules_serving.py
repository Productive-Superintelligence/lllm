"""Canned model replies for docs/serving.md (see docs_model.py)."""

from __future__ import annotations

import json

from docs_model import asks, rule


@rule(asks("route this ticket to billing, bugs, or general", "charged twice"))
def _route_double_charge(request):
    return json.dumps({"queue": "billing", "reason": "Duplicate charge."})
