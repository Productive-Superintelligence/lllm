"""Regression tests for native runtime fixes found while rewriting the docs."""

import json

import pytest
from pydantic import BaseModel

pytest.importorskip("litellm", reason="these tests drive the native runtime")

from lllm import Prompt, Tactic, tool  # noqa: E402
from lllm.runtimes.native import Dialog, Role  # noqa: E402
from lllm.runtimes.native.core.dialog import DefaultContextManager  # noqa: E402
from lllm.testing import ScriptedModel, ToolCall  # noqa: E402


# -- 1: the agent loop's working copies stay out of the dialog tree --------


class Rating(BaseModel):
    stars: int


@tool(description="Look up a word")
def define(word: str) -> str:
    return f"{word}: damaged by pressure"


def test_respond_keeps_working_copies_out_of_the_dialog_tree():
    rate = Prompt(
        path="fix/rate",
        prompt="Rate 1-5: {review}",
        format=Rating,
        function_list=[define],
    )
    agent = Tactic.quick(system_prompt="You rate reviews.", model="gpt-4o-mini")
    agent.open("rate")
    agent.receive_prompt(rate, {"review": "Crushed box."})
    replies = [ToolCall("define", {"word": "crushed"}), "Four.", '{"stars": 4}']
    with ScriptedModel(replies):
        session = agent.respond(return_session=True)

    assert session.exception_retries_count == 1
    assert len(session.tool_calls) == 1
    dialog = agent.current_dialog
    # system, prompt, tool call, tool result, accepted reply; no repairs
    assert [m.role for m in dialog.messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.TOOL_CALL,
        Role.TOOL,
        Role.ASSISTANT,
    ]
    assert dialog.children == []
    assert dialog.tree_node.children_ids == []
    assert dialog.to_dict()["children"] == []
    assert dialog.tree_overview().count("\n") == 0

    agent.fork("rate", "retry")
    assert dialog.children == [agent.dialogs["retry"]]
    assert dialog.tree_node.children_ids == [agent.dialogs["retry"].dialog_id]


def test_context_manager_copies_stay_out_of_the_dialog_tree():
    agent = Tactic.quick(
        system_prompt="You take notes.",
        model="gpt-4o-mini",
    )
    agent.context_manager = DefaultContextManager("gpt-4o-mini", max_tokens=5_030)
    agent.open("notes")
    with ScriptedModel(["Noted."]) as model:
        for day in ("Monday", "Tuesday", "Wednesday", "Thursday"):
            agent.receive(f"Note: on {day} we shipped a small fix.")
            agent.respond()

    assert len(model.requests[-1].messages) < 2 * 4 + 1  # it did trim
    assert agent.current_dialog.children == []
    assert len(agent.current_dialog.messages) == 9


def test_saved_dialogs_grow_linearly_with_turns():
    def saved_size(turns: int) -> int:
        agent = Tactic.quick(system_prompt="You echo.", model="gpt-4o-mini")
        agent.open("echo")
        with ScriptedModel(["ok"]):
            for turn in range(turns):
                agent.receive(f"turn {turn}")
                agent.respond()
        return len(json.dumps(agent.current_dialog.to_dict()))

    assert saved_size(40) < 2.2 * saved_size(20)


def test_working_copy_names_its_dialog_as_parent():
    dialog = Dialog(owner="agent")
    dialog.put_text("You help.", name="system", role=Role.SYSTEM)
    copy = dialog._copy()

    assert copy.parent is dialog
    assert copy.tree_node.parent_id == dialog.dialog_id
    assert copy.depth == 1
    assert [m.content for m in copy.messages] == ["You help."]
    assert dialog.children == []
