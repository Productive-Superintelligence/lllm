"""The tactic completion log line shows the cost in dollars."""

import logging
import re

import pytest

pytest.importorskip("litellm", reason="the tactic calls a native agent")

from lllm import Tactic  # noqa: E402
from lllm.testing import ScriptedModel  # noqa: E402


class Summarize(Tactic):
    name = "summarize"
    agent_group = ["writer"]

    def call(self, text: str) -> str:
        writer = self.agents["writer"]
        writer.open("summary")
        writer.receive(f"Summarize: {text}")
        return writer.respond().content


def test_the_completion_line_formats_the_cost_in_dollars(caplog):
    summarize = Summarize({
        "global": {"model_name": "gpt-4o-mini"},
        "agent_configs": [{"name": "writer", "system_prompt": "Be brief."}],
    })

    with caplog.at_level(logging.INFO, logger="lllm.runtimes.native.core.tactic"):
        with ScriptedModel(["Short."]):
            session = summarize("A long text.", session_name="run-1", return_session=True)

    assert session.total_cost.cost > 0
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == "Tactic 'summarize' started — session_name=run-1"
    assert re.fullmatch(
        r"Tactic 'summarize' completed — cost=\$\d+\.\d{6} agent_calls=1", messages[-1]
    ), messages[-1]
    assert f"cost=${session.total_cost.cost:.6f}" in messages[-1]
