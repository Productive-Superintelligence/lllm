"""Canned model replies for docs/examples/http-service.md (see docs_model.py).

The policy checker runs inside the served app (in a server thread); the
support assistant runs in the client and calls it through a RemoteTactic.
"""

from __future__ import annotations

import json

from docs_model import asks, rule, tool_result
from lllm.testing import ToolCall

LEAKY = "Your export is ready: https://admin.internal/exp/88"
CLEAN = "Your export is ready. Download it from Settings > Exports."


@rule(asks("check this email message against the data policy"))
def _policy_check(request):
    message = request.text.split("Message:", 1)[1]
    problems = ["Links to an internal admin page."] if ".internal" in message else []
    return json.dumps({"allowed": not problems, "problems": problems})


@rule(lambda r: "policy_check" in r.tools and not r.tool_results())
def _assistant_first_check(request):
    return ToolCall("policy_check", {"text": LEAKY, "channel": "email"})


@rule(lambda r: "policy_check" in r.tools and "allowed=False" in tool_result(r))
def _assistant_second_check(request):
    return ToolCall("policy_check", {"text": CLEAN, "channel": "email"})


@rule(lambda r: "policy_check" in r.tools and "allowed=True" in tool_result(r))
def _assistant_answer(request):
    return CLEAN
