"""Canned model replies for docs/examples/review-loop.md (see docs_model.py)."""

from __future__ import annotations

import json

from docs_model import asks, rule

FIRST_DRAFT = """\
Subject: API v1 shutdown

Hello,

We are turning off the gw-legacy gateway, so API v1 stops working on
March 31, 2027. Please move to API v2.

Thanks,
The Platform team"""

REVISED_DRAFT = """\
Subject: API v1 retires on March 31, 2027

Hello,

API v1 will stop working on March 31, 2027. Please move to API v2 before
then. The migration guide walks through each change:
https://example.com/v2-migration

Thanks,
The Platform team"""


@rule(asks("write a short email to customers about this change"))
def _first_draft(request):
    return FIRST_DRAFT


@rule(asks("revise the email. fix these issues"))
def _revised_draft(request):
    return REVISED_DRAFT


@rule(asks("review this customer email against the style guide"))
def _review(request):
    email = request.text.split("Email:", 1)[1]
    issues = []
    if "gw-legacy" in email:
        issues.append("Uses the internal name gw-legacy; customers do not know it.")
    if "v2-migration" not in email:
        issues.append("Does not link the migration guide.")
    return json.dumps({"approved": not issues, "issues": issues})
