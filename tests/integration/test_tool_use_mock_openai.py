import pytest

from lllm.core.const import Roles
from lllm.core.models import Function, Prompt
from lllm.invokers.litellm import LiteLLMInvoker
from tests.helpers.agent_utils import make_agent
from tests.helpers.mock_openai import MockOpenAIClient, text_completion, tool_call_completion


def test_tool_use_flow_with_mock_litellm(monkeypatch, log_config):
    # Arrange tool
    calls = []

    def get_weather(location: str, unit: str = "celsius"):
        calls.append((location, unit))
        return f"{location}:{unit}"

    tool = Function(
        name="get_weather",
        description="Return weather for a location",
        properties={
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        required=["location"],
    )
    tool.link_function(get_weather)

    system_prompt = Prompt(path="mock/system", prompt="Use tools when appropriate.")
    task_prompt = Prompt(
        path="mock/task",
        prompt="Please check the weather in {city}.",
        function_list=[tool],
    )

    # Script OpenAI responses: first trigger tool call, then final response
    scripts = [
        tool_call_completion(
            "get_weather", {"location": "Tokyo", "unit": "celsius"}
        ),
        text_completion("Weather retrieved."),
    ]

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def fake_openai_client(*args, **kwargs):
        return MockOpenAIClient(scripts.copy())

    import openai

    monkeypatch.setattr(openai, "OpenAI", fake_openai_client)

    invoker = LiteLLMInvoker({})
    agent = make_agent(system_prompt, invoker, log_config)

    dialog = agent.init_dialog()
    dialog.send_message(task_prompt, {"city": "Tokyo"})

    response, dialog, interrupts = agent.call(dialog)

    assert response.content == "Weather retrieved."
    assert calls == [("Tokyo", "celsius")]
    assert len(interrupts) == 1
    assert interrupts[0].result == "Tokyo:celsius"
