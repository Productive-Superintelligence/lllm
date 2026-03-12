from pathlib import Path

import pytest

from lllm.core.models import Function, Prompt
from lllm.invokers.litellm import LiteLLMInvoker
from tests.helpers.agent_utils import make_agent
from tests.helpers.mock_openai import MockOpenAIClient, load_recorded_completions


RECORDING_PATH = Path(__file__).parent / "recordings" / "sample_tool_call.json"


def test_tool_use_with_recorded_payload_with_litellm(monkeypatch, log_config):
    calls = []

    def get_weather(location: str, unit: str = "celsius"):
        calls.append((location, unit))
        return f"{location}:{unit}"

    tool = Function(
        name="get_weather",
        description="Return weather for a location",
        properties={
            "location": {"type": "string"},
            "unit": {"type": "string"},
        },
        required=["location"],
    )
    tool.link_function(get_weather)

    system_prompt = Prompt(path="recorded/system", prompt="Use tools.")
    task_prompt = Prompt(
        path="recorded/task",
        prompt="Check weather in {city}.",
        function_list=[tool],
    )

    scripts = load_recorded_completions(RECORDING_PATH)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def fake_client(*args, **kwargs):
        return MockOpenAIClient(scripts.copy())

    import openai

    monkeypatch.setattr(openai, "OpenAI", fake_client)

    invoker = LiteLLMInvoker({})
    agent = make_agent(system_prompt, invoker, log_config)

    dialog = agent.init_dialog()
    dialog.send_message(task_prompt, {"city": "Berlin"})

    response, dialog, interrupts = agent.call(dialog)
    assert response.content == "Recorded weather response."
    assert calls == [("Berlin", "celsius")]
    assert len(interrupts) == 1
