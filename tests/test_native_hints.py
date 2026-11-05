"""Inference hints: session identity and serving intent on every agent call."""

import pytest
from pydantic import ValidationError

from lllm.runtimes.native import Agent, InferenceHints, Message, Prompt, Role
from lllm.runtimes.native.core.const import InvokeResult
from lllm.runtimes.native.core.hints import merge_request_fields
from lllm.runtimes.native.core.native_config import AgentSpec


class RecordingInvoker:
    """Fake invoker that records the model args of every call."""

    def __init__(self):
        self.calls = []

    def call(self, dialog, model, model_args=None, **kwargs):
        self.calls.append(dict(model_args or {}))
        return InvokeResult(
            message=Message(role=Role.ASSISTANT, content="ok", name="assistant")
        )


def make_agent(hints, **model_args):
    invoker = RecordingInvoker()
    agent = Agent(
        name="planner",
        system_prompt=Prompt(path="hints/system", prompt="You plan."),
        model="hosted_vllm/qwen",
        llm_invoker=invoker,
        model_args=model_args,
        inference_hints=hints,
    )
    return agent, invoker


def test_dynamo_profile_sends_session_headers_and_agent_hints():
    hints = InferenceHints(
        profile="dynamo", priority=5, strict_priority=1, speculative_prefill=True
    )

    fields = hints.request_fields(
        dialog_id="d-1", parent_dialog_id="d-0", model_args={"max_tokens": 512}
    )

    assert fields == {
        "extra_headers": {
            "x-dynamo-session-id": "d-1",
            "x-dynamo-parent-session-id": "d-0",
        },
        "extra_body": {
            "nvext": {
                "agent_hints": {
                    "priority": 5,
                    "strict_priority": 1,
                    "osl": 512,
                    "speculative_prefill": True,
                }
            }
        },
    }


def test_vllm_profile_inverts_priority_and_salts_the_prefix_cache():
    fields = InferenceHints(
        profile="vllm", priority=3, cache_salt="tenant-a", session_prefix="run-42"
    ).request_fields(dialog_id="d-1")

    assert fields == {
        "extra_headers": {"x-session-id": "run-42:d-1"},
        "extra_body": {"priority": -3, "cache_salt": "tenant-a"},
    }


def test_sglang_and_openai_profiles():
    assert InferenceHints(profile="sglang", priority=2).request_fields(
        dialog_id="d"
    ) == {"extra_headers": {"x-session-id": "d"}, "extra_body": {"priority": 2}}
    # The generic profile never adds body fields that strict APIs could reject.
    assert InferenceHints(priority=9, osl=10).request_fields(dialog_id="d") == {
        "extra_headers": {"x-session-id": "d"}
    }


def test_sessions_can_be_turned_off_and_unset_fields_are_omitted():
    assert InferenceHints(profile="dynamo", session=False).request_fields(
        dialog_id="d"
    ) == {}
    assert InferenceHints(profile="vllm").request_fields() == {}


def test_final_request_marks_the_dynamo_session_finished():
    fields = InferenceHints(profile="dynamo").final_request_fields("d-1")

    assert fields["extra_headers"] == {
        "x-dynamo-session-id": "d-1",
        "x-dynamo-session-final": "true",
    }


def test_explicit_model_args_win_over_hints():
    merged = merge_request_fields(
        {"extra_headers": {"x-session-id": "mine"}, "temperature": 0},
        {
            "extra_headers": {"x-session-id": "d", "x-parent-session-id": "p"},
            "extra_body": {"priority": 1},
        },
    )

    assert merged == {
        "extra_headers": {"x-session-id": "mine", "x-parent-session-id": "p"},
        "extra_body": {"priority": 1},
        "temperature": 0,
    }


def test_hint_validation():
    with pytest.raises(ValidationError):
        InferenceHints(profile="triton")
    with pytest.raises(ValidationError):
        InferenceHints(strict_priority=-1)
    with pytest.raises(ValidationError):
        InferenceHints(cache_salt=" ")
    with pytest.raises(ValidationError):
        InferenceHints(unknown=True)


def test_agent_sends_one_session_per_dialog_and_links_forks():
    agent, invoker = make_agent(
        InferenceHints(profile="dynamo", priority=4), max_tokens=128
    )
    agent.open("main")
    agent.receive("first")
    agent.respond()
    agent.receive("second")
    agent.respond()
    main_id = agent.dialogs["main"].tree_node.dialog_id

    agent.fork("main", "retry")
    agent.respond()
    retry_id = agent.dialogs["retry"].tree_node.dialog_id

    first, second, forked = invoker.calls
    assert first["extra_headers"] == {"x-dynamo-session-id": main_id}
    assert second["extra_headers"] == first["extra_headers"]
    assert forked["extra_headers"] == {
        "x-dynamo-session-id": retry_id,
        "x-dynamo-parent-session-id": main_id,
    }
    assert first["extra_body"] == {
        "nvext": {"agent_hints": {"priority": 4, "osl": 128}}
    }
    assert first["max_tokens"] == 128


def test_agents_without_hints_send_nothing_extra():
    agent, invoker = make_agent(None, temperature=0)
    agent.open("main")
    agent.receive("hello")
    agent.respond()

    assert invoker.calls == [{"temperature": 0}]


def test_agent_config_accepts_inference_hints():
    spec = AgentSpec.from_config(
        "planner",
        {
            "model_name": "hosted_vllm/qwen",
            "system_prompt": "You plan.",
            "inference_hints": {"profile": "sglang", "priority": 7},
        },
    )

    assert spec.inference_hints == InferenceHints(profile="sglang", priority=7)
    assert "inference_hints" not in spec.model_args
