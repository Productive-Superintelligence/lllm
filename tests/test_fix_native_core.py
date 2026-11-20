"""Regression tests for native runtime fixes found while rewriting the docs."""

import base64
import json
from collections import namedtuple
from decimal import Decimal
from operator import itemgetter
from typing import Any, ClassVar, Dict, List, Literal, Optional
from unittest.mock import MagicMock

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    computed_field,
    model_validator,
)

pytest.importorskip("litellm", reason="these tests drive the native runtime")

from lllm import ParseError, Prompt, Tactic, tool  # noqa: E402
from lllm.runtimes.native import (  # noqa: E402
    DefaultContextManager,
    Dialog,
    InvokeCost,
    Modalities,
    Role,
)
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
    agent = Tactic.quick(system_prompt="You take notes.", model="gpt-4o-mini")
    agent.context_manager = Exact("gpt-4o-mini", max_tokens=40)
    agent.open("notes")
    with ScriptedModel(["Noted."]) as model:
        for day in ("Monday", "Tuesday", "Wednesday", "Thursday"):
            agent.receive(f"Note: on {day} we shipped a small fix.")
            agent.respond()

    dialog = agent.current_dialog
    assert len(dialog.messages) == 9
    # The last call answered the first 8 messages; trimming sent fewer.
    assert len(model.requests[-1].messages) < len(dialog.messages) - 1
    assert dialog.children == []


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


# -- 2: the context manager's budget is the context window -----------------


def _dialog(*texts: str) -> Dialog:
    dialog = Dialog(owner="agent")
    dialog.put_text("You help.", name="system", role=Role.SYSTEM)
    for text in texts:
        dialog.put_text(text)
    return dialog


def test_default_context_manager_budget_is_the_context_window():
    assert DefaultContextManager("gpt-4o-mini").max_tokens == 128_000
    assert DefaultContextManager("gpt-4o-mini", max_tokens=9_000).max_tokens == 9_000

    # About 15,000 tokens: more than gpt-4o-mini's output limit (16,384)
    # leaves after the safety buffer, far less than its 128,000 window.
    dialog = _dialog("word " * 15_000, "What now?")
    assert DefaultContextManager("gpt-4o-mini")(dialog) is dialog


def _forget_window(model: str) -> None:
    from lllm.runtimes.native.core import dialog as dialog_module

    dialog_module._CONTEXT_WINDOWS.pop(model, None)
    dialog_module._UNKNOWN_WINDOWS.discard(model)


def test_an_unknown_window_is_not_trimmed_and_warned_about_once(caplog):
    model = "openai/fix-native-core-unknown-model"
    _forget_window(model)
    manager = DefaultContextManager(model)
    dialog = _dialog("word " * 10_000, "What now?")  # over any default window

    with caplog.at_level("WARNING", logger="lllm.runtimes.native.core.dialog"):
        assert manager.max_tokens is None
        assert manager(dialog) is dialog
        assert DefaultContextManager(model).max_tokens is None
    warnings = [r for r in caplog.records if "context window" in r.getMessage()]
    assert len(warnings) == 1
    assert "set max_tokens" in warnings[0].getMessage().lower()


def _register(model: str, **limits: int) -> None:
    import litellm

    litellm.register_model(
        {
            model: {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                **limits,
            }
        }
    )


def test_registered_models_give_their_window_as_max_input_tokens():
    import litellm

    only_max, with_input = "fixcore/only-max-tokens", "fixcore/with-input"
    for model in (only_max, with_input):
        _forget_window(model)
    try:
        # max_tokens alone may be an output limit, so it is not a window.
        _register(only_max, max_tokens=8192)
        assert DefaultContextManager(only_max).max_tokens is None
        # Unknown at first; found once registered (None is not remembered).
        assert DefaultContextManager(with_input).max_tokens is None
        _register(with_input, max_input_tokens=32_000, max_tokens=4096)
        assert DefaultContextManager(with_input).max_tokens == 32_000
    finally:
        for model in (only_max, with_input):
            litellm.model_cost.pop(model, None)
            _forget_window(model)


def _fake_litellm(monkeypatch, provider: str, info: Dict[str, Any]) -> list:
    """Make LiteLLM know one model of ``provider`` by its unprefixed name."""
    import litellm

    lookups = []

    def get_llm_provider(model, *args, **kwargs):
        return model.split("/", 1)[1], provider, None, None

    def get_model_info(model, custom_llm_provider=None, **kwargs):
        lookups.append((model, custom_llm_provider))
        if model.startswith("huggingface/"):  # the hub URL 404s on the prefix
            raise Exception(f"{model} not found")
        return dict(info, litellm_provider=provider)

    monkeypatch.setattr(litellm, "get_llm_provider", get_llm_provider)
    monkeypatch.setattr(litellm, "get_model_info", get_model_info)
    return lookups


def test_a_hugging_face_window_is_looked_up_without_the_prefix(monkeypatch):
    model = "huggingface/fixcore-org/fixcore-model"
    _forget_window(model)
    # LiteLLM gives the hub's max_position_embeddings as max_tokens.
    lookups = _fake_litellm(monkeypatch, "huggingface", {"max_tokens": 32_768})
    try:
        assert DefaultContextManager(model).max_tokens == 32_768
    finally:
        _forget_window(model)
    assert lookups[0] == ("fixcore-org/fixcore-model", "huggingface")


def test_max_tokens_alone_is_an_output_limit_for_other_providers(monkeypatch):
    model = "gradient_ai/fixcore-model"
    _forget_window(model)
    _fake_litellm(monkeypatch, "gradient_ai", {"max_tokens": 4_096})
    try:
        assert DefaultContextManager(model).max_tokens is None
    finally:
        _forget_window(model)


def test_a_model_under_an_unknown_provider_is_found_by_its_whole_name(
    monkeypatch,
):
    import litellm

    model = "fixcore/custom-provider"
    _forget_window(model)
    real, found = litellm.get_model_info, []

    def get_model_info(name, *args, **kwargs):
        info = real(name, *args, **kwargs)
        found.append((name, kwargs.get("custom_llm_provider")))
        return info

    monkeypatch.setattr(litellm, "get_model_info", get_model_info)
    litellm.register_model(
        {
            model: {
                "litellm_provider": "fixcore",
                "mode": "chat",
                "max_input_tokens": 2_000,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            }
        }
    )
    try:
        assert DefaultContextManager(model).max_tokens == 2_000
        assert found == [(model, None)]  # the whole-name lookup found it
    finally:
        litellm.model_cost.pop(model, None)
        _forget_window(model)


# -- 3: a message that partly fits keeps the part the marker says ----------


MARKER = "[...earlier content truncated...]\n"
CM_LOGGER = "lllm.runtimes.native.core.dialog"


class Exact(DefaultContextManager):
    """No safety buffer: ``max_tokens`` is exactly the budget."""

    SAFETY_BUFFER = 0


def test_partly_kept_message_keeps_its_end_after_the_marker():
    long = "BEGIN " + " ".join(f"w{i}" for i in range(400)) + " END"
    dialog = _dialog(long, "Short question?")
    trimmed = Exact("gpt-4o-mini", max_tokens=200)(dialog)

    border, last = trimmed.messages[1:]
    assert last.content == "Short question?"
    assert border.content.startswith(MARKER)
    assert border.content.endswith(" END")
    assert "BEGIN" not in border.content
    assert long.endswith(border.content.removeprefix(MARKER))
    assert dialog.messages[1].content == long  # the dialog is never changed


def _photo(size: int) -> str:
    """Base64 data that put_image() takes for a PNG of ``size`` bytes."""
    import random

    data = b"\x89PNG\r\n\x1a\n" + random.Random(size).randbytes(size)
    return base64.b64encode(data).decode()


def _tokens(manager: DefaultContextManager, message: Any) -> int:
    return manager._tokens(manager._to_raw(message))


def _text_tokens(manager: DefaultContextManager, text: str) -> int:
    return manager._count([{"role": "user", "content": text, "name": "user"}])


IMAGE_TOKENS = DefaultContextManager.IMAGE_TOKENS


def _image(
    width: int, height: int, fmt: str = "PNG", mode: str = "RGB", **save: Any
) -> bytes:
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    data = io.BytesIO()
    Image.new(mode, (width, height)).save(data, format=fmt, **save)
    return data.getvalue()


def test_a_large_photo_has_its_shortest_side_scaled_to_768():
    late = _image(1920, 1080, fmt="JPEG", comment=b"x" * 60_000)
    from lllm.runtimes.native.core.dialog import _IMAGE_HEAD

    # Its size marker lies past the first read, _IMAGE_HEAD base64 characters.
    assert late.find(b"\xff\xc0") > _IMAGE_HEAD * 3 // 4
    manager = DefaultContextManager("gpt-4o-mini")

    for data in (_image(1920, 1080), late):
        dialog = _dialog()
        dialog.put_image(base64.b64encode(data).decode())
        # 1920x1080 scales to 1365x768: 3 x 2 tiles, 85 + 170 * 6.
        assert _tokens(manager, dialog.messages[1]) == _text_tokens(manager, "") + 1105


def test_images_are_sized_from_their_header_with_openai_high_detail():
    dialog = _dialog()
    dialog.put_image(base64.b64encode(_image(1024, 768)).decode(), caption="A")
    dialog.put_image(_photo(360_000))  # a PNG signature, then noise
    dialog.put_text("What is in them?")
    manager = DefaultContextManager("gpt-4o-mini")
    photo, noise = dialog.messages[1:3]

    assert _tokens(manager, photo) == _text_tokens(manager, "A") + 765
    assert _tokens(manager, noise) == _text_tokens(manager, "") + IMAGE_TOKENS
    assert manager(dialog) is dialog


def test_a_bmp_from_a_path_is_sized_too(tmp_path):
    path = tmp_path / "wide.bmp"
    path.write_bytes(_image(3000, 1000, fmt="BMP", mode="1"))
    dialog = _dialog()
    dialog.put_image(path)
    manager = DefaultContextManager("gpt-4o-mini")

    # 3000x1000 fits in 2048x683: 4 x 2 tiles of 512 px, 85 + 170 * 8.
    assert _tokens(manager, dialog.messages[1]) == _text_tokens(manager, "") + 1445


def _with_parts(*parts: Dict[str, Any]) -> Any:
    from lllm.runtimes.native import Message

    content = [{"type": "text", "text": "Look at this."}, *parts]
    return Message(role=Role.USER, name="user", content=content)


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "image_url", "image_url": "https://example.com/a.png"},
        {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
        {"type": "file", "file": {"file_id": "file-1"}},
    ],
    ids=["http_url", "string_url", "input_audio", "unknown"],
)
def test_other_parts_count_as_image_tokens_without_network(part, monkeypatch):
    import socket

    attempts = []  # recorded: a raised error alone could be swallowed

    def no_network(*args, **kwargs):
        attempts.append(args)
        raise OSError("counting reached the network")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    manager = DefaultContextManager("gpt-4o-mini")

    tokens = _tokens(manager, _with_parts(part))
    assert tokens == _text_tokens(manager, "Look at this.") + IMAGE_TOKENS
    assert attempts == []


def _unsizable_image(caption: str) -> Any:
    from lllm.runtimes.native import Message

    return Message(
        role=Role.USER,
        name="user",
        content="!!not base64!!",
        modality=Modalities.IMAGE,
        metadata={"caption": caption},
    )


def test_an_unsizable_image_counts_as_image_tokens():
    dialog = _dialog("What is in it?")
    dialog.append(_unsizable_image("A photo"))
    manager = DefaultContextManager("gpt-4o-mini")

    tokens = _tokens(manager, dialog.messages[-1])
    assert tokens == _text_tokens(manager, "A photo") + IMAGE_TOKENS
    assert manager(dialog) is dialog  # counting the dialog does not fail


def test_trimming_drops_an_unsizable_image_whole(caplog):
    dialog = _dialog()
    dialog.append(_unsizable_image("Q3 revenue"))
    dialog.put_text("What does it show?")
    with caplog.at_level("WARNING", logger=CM_LOGGER):
        trimmed = Exact("gpt-4o-mini", max_tokens=300)(dialog)

    assert [m.modality for m in trimmed.messages] == [Modalities.TEXT] * 2
    assert "dropped 1 image" in caplog.text


def test_tool_call_arguments_are_counted():
    dialog = _dialog("Find notes about LLLM.")
    _tool_round(dialog, "Found 3 notes.")
    dialog.messages[2].function_calls[0].arguments = {"query": "lllm " * 400}
    manager = DefaultContextManager("gpt-4o-mini")

    assert _tokens(manager, dialog.messages[2]) > 400  # the content is short


class Counting(Exact):
    """Records how many messages each count gets; may refuse batches."""

    def __init__(self, *args: Any, refuse_batches: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.counted: List[int] = []
        self.refuse_batches = refuse_batches

    def _count(self, raw_msgs: list) -> int:
        self.counted.append(len(raw_msgs))
        if self.refuse_batches and len(raw_msgs) > 1:
            raise ValueError("cannot count a batch")
        return super()._count(raw_msgs)


def _per_message_sum(dialog: Dialog) -> int:
    manager = DefaultContextManager("gpt-4o-mini")
    return sum(manager._tokens(manager._to_raw(m)) for m in dialog.messages)


def test_a_dialog_that_fits_by_one_count_is_not_trimmed():
    dialog = _dialog(*(f"ok {n}" for n in range(60)))
    raws = [DefaultContextManager("gpt-4o-mini")._to_raw(m) for m in dialog.messages]
    batch = DefaultContextManager("gpt-4o-mini")._count(raws)
    # Each count adds 3 reply-priming tokens, so the per-message sum is more.
    assert _per_message_sum(dialog) > batch

    manager = Counting("gpt-4o-mini", max_tokens=batch)
    assert manager(dialog) is dialog
    assert manager.counted == [len(dialog.messages)]  # one call, all of it


def test_when_a_batch_cannot_be_counted_the_per_message_sum_decides():
    dialog = _dialog(*(f"ok {n}" for n in range(20)))
    total = _per_message_sum(dialog)

    fits = Counting("gpt-4o-mini", max_tokens=total, refuse_batches=True)
    assert fits(dialog) is dialog
    over = Counting("gpt-4o-mini", max_tokens=total - 1, refuse_batches=True)
    assert over(dialog) is not dialog  # trimmed, and nothing raised


def test_the_newest_message_is_sent_whole_even_over_budget(caplog):
    long = "word " * 500
    dialog = _dialog("An older note.", long)
    with caplog.at_level("WARNING", logger=CM_LOGGER):
        trimmed = Exact("gpt-4o-mini", max_tokens=60)(dialog)
    assert [m.content for m in trimmed.messages] == ["You help.", long]
    assert "newest" in caplog.text

    dialog = _dialog("An older note.")
    dialog.put_image(_photo(1_000), caption="A photo")
    trimmed = Exact("gpt-4o-mini", max_tokens=40)(dialog)
    assert [m.modality for m in trimmed.messages] == [
        Modalities.TEXT,
        Modalities.IMAGE,
    ]
    assert trimmed.messages[1].content == dialog.messages[2].content


def test_a_dropped_image_is_dropped_whole_and_logged(caplog):
    dialog = _dialog()
    dialog.put_image(base64.b64encode(_image(1024, 768)).decode(), caption="Q3")
    dialog.put_text("word " * 50)
    manager = Exact("gpt-4o-mini", max_tokens=300)
    head, image, newest = (_tokens(manager, m) for m in dialog.messages)
    # Enough is left at the image to cut a text message (over 50 tokens),
    # but not to keep the image, so only the guard keeps it from being cut.
    assert 50 < 300 - head - newest < image
    with caplog.at_level("WARNING", logger=CM_LOGGER):
        trimmed = manager(dialog)

    assert [m.modality for m in trimmed.messages] == [Modalities.TEXT] * 2
    assert "dropped 1 image" in caplog.text


def _tool_round(dialog: Dialog, *results: str) -> None:
    """A tool-call reply and one tool result per call, as respond() adds."""
    from lllm.runtimes.native import FunctionCall, Message

    calls = [
        FunctionCall(id=f"call_{i}", name="lookup_order", arguments={"n": i})
        for i in range(len(results))
    ]
    dialog.append(
        Message(
            role=Role.TOOL_CALL,
            name="assistant",
            content="Tool calls:\n\n0. lookup_order",
            function_calls=calls,
        )
    )
    for call, result in zip(calls, results):
        dialog.put_text(
            result,
            name=call.name,
            role=Role.TOOL,
            metadata={"tool_call_id": call.id},
        )


def _roles(dialog: Dialog) -> list:
    return [message.role for message in dialog.messages]


def test_a_tool_call_is_kept_or_dropped_with_its_results():
    dialog = _dialog("Where are orders 1 and 2? I need them soon. " * 10)
    _tool_round(dialog, "Order 1 shipped. " * 20, "Order 2 is packed. " * 20)
    dialog.put_text("And when do they arrive?")
    manager = Exact("gpt-4o-mini", max_tokens=10_000)
    head, _, call, *results, newest = dialog.messages
    tokens = {id(m): manager._count([manager._to_raw(m)]) for m in dialog.messages}

    # Room for the newest message and both results, but not for their call.
    budget = sum(tokens[id(m)] for m in (head, newest, *results))
    budget += tokens[id(call)] // 2
    trimmed = Exact("gpt-4o-mini", max_tokens=budget)(dialog)
    assert _roles(trimmed) == [Role.SYSTEM, Role.USER]

    budget += tokens[id(call)]  # now the whole round fits
    trimmed = Exact("gpt-4o-mini", max_tokens=budget)(dialog)
    assert _roles(trimmed) == [
        Role.SYSTEM,
        Role.TOOL_CALL,
        Role.TOOL,
        Role.TOOL,
        Role.USER,
    ]


def test_a_newest_tool_result_keeps_its_call():
    dialog = _dialog("Where is order 1?")
    _tool_round(dialog, "Order 1 shipped on Monday. " * 30)
    trimmed = Exact("gpt-4o-mini", max_tokens=60)(dialog)

    assert _roles(trimmed) == [Role.SYSTEM, Role.TOOL_CALL, Role.TOOL]


def _trims(manager: DefaultContextManager, words: int) -> bool:
    dialog = _dialog("word " * words, "What now?")  # about words + 20 tokens
    return manager(dialog) is not dialog


def test_the_default_buffer_is_5000_tokens_or_half_a_smaller_budget():
    def default(max_tokens: int) -> DefaultContextManager:
        return DefaultContextManager("gpt-4o-mini", max_tokens=max_tokens)

    # 16,000 left after 5,000; an uncapped half or quarter would leave less.
    assert not _trims(default(21_000), 15_880)
    # 4,000 left after half; a quarter would leave 6,000.
    assert _trims(default(8_000), 4_990)
    assert not _trims(default(8_000), 3_900)
    # 800 left after half: small budgets still work.
    assert not _trims(default(1_600), 700)


def test_a_safety_buffer_that_is_set_is_used_as_is():
    class Roomy(DefaultContextManager):
        SAFETY_BUFFER = 6_000  # more than half of 8,000

    # 2,000 left; a value capped at half the budget would leave 4,000.
    assert _trims(Roomy("gpt-4o-mini", max_tokens=8_000), 2_990)

    roomy = DefaultContextManager("gpt-4o-mini", max_tokens=8_000)
    roomy.SAFETY_BUFFER = 6_000
    assert _trims(roomy, 2_990)

    # On the class itself, the one-line way to set it for config-built agents.
    default = DefaultContextManager.SAFETY_BUFFER
    try:
        DefaultContextManager.SAFETY_BUFFER = 6_000
        assert _trims(DefaultContextManager("gpt-4o-mini", max_tokens=8_000), 2_990)
    finally:
        DefaultContextManager.SAFETY_BUFFER = default
    assert DefaultContextManager.SAFETY_BUFFER == 5_000  # as in 0.4.0


# -- 4: a Pydantic config keeps its aliased ``global`` section ---------------


class AgentEntry(BaseModel):
    name: str
    system_prompt: str


class EchoConfig(BaseModel):
    global_: Dict[str, Any] = Field(default_factory=dict, alias="global")
    agent_configs: List[AgentEntry]


class Echo(Tactic):
    name = "fix_echo"
    agent_group = ["echo"]

    def call(self, text: str) -> str:
        echo = self.agents["echo"]
        echo.open("echo")
        echo.receive(text)
        return echo.respond().content


def test_pydantic_config_keeps_its_global_section():
    config = EchoConfig.model_validate(
        {
            "global": {"model_name": "gpt-4o-mini", "max_llm_recall": 2},
            "agent_configs": [{"name": "echo", "system_prompt": "You echo."}],
        }
    )
    tactic = Echo(config)

    echo = tactic.agents["echo"]
    assert (echo.model, echo.max_llm_recall) == ("gpt-4o-mini", 2)
    with ScriptedModel(["hello"]):
        assert tactic("hello") == "hello"


GLOBAL = {"model_name": "gpt-4o-mini", "max_llm_recall": 2}


@pytest.mark.parametrize("by_alias", [False, True], ids=["dump", "serialize_by_alias"])
def test_a_config_with_an_alias_generator_is_read_by_field_name(by_alias):
    from pydantic.alias_generators import to_camel

    camel = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, serialize_by_alias=by_alias
    )

    class CamelEntry(BaseModel):
        model_config = camel
        name: str
        system_prompt: str

    class CamelConfig(BaseModel):
        model_config = camel
        global_: Dict[str, Any] = Field(default_factory=dict, alias="global")
        agent_configs: List[CamelEntry]

    config = CamelConfig.model_validate(
        {
            "global": GLOBAL,
            "agentConfigs": [{"name": "echo", "systemPrompt": "You echo."}],
        }
    )
    echo = Echo(config).agents["echo"]
    assert (echo.model, echo.max_llm_recall, echo.system_prompt.prompt) == (
        "gpt-4o-mini",
        2,
        "You echo.",
    )


@pytest.mark.parametrize(
    ("alias", "key"),
    [
        ({"validation_alias": "global"}, "global"),
        ({"validation_alias": AliasChoices("defaults", "global")}, "global"),
        ({"validation_alias": AliasPath("global")}, "global"),
        ({"validation_alias": AliasChoices("defaults", AliasPath("global"))}, "global"),
        ({"serialization_alias": "global"}, "global_"),
    ],
    ids=["validation_alias", "alias_choices", "alias_path", "path_choice", "serialization"],
)
def test_the_field_aliased_global_is_the_global_section(alias, key):
    class Config(BaseModel):
        global_: Dict[str, Any] = Field(default_factory=dict, **alias)
        agent_configs: List[AgentEntry]

    config = Config.model_validate(
        {key: GLOBAL, "agent_configs": [{"name": "echo", "system_prompt": "Hi."}]}
    )
    echo = Echo(config).agents["echo"]
    assert (echo.model, echo.max_llm_recall) == ("gpt-4o-mini", 2)


class TrimSettings(BaseModel):
    type: Optional[str] = "default"
    max_tokens: Optional[int] = None


class TeamEntry(BaseModel):
    name: str
    system_prompt: str
    model_name: Optional[str] = None
    context_manager: Optional[TrimSettings] = None


class TeamConfig(BaseModel):
    global_: Dict[str, Any] = Field(default_factory=dict, alias="global")
    agent_configs: List[TeamEntry]
    max_workers: Optional[int] = None


class Team(Tactic):
    name = "fix_team"
    agent_group = ["writer", "editor", "archivist"]


def test_unset_none_fields_fall_back_to_global_and_set_ones_apply():
    config = TeamConfig.model_validate(
        {
            "global": {
                "model_name": "gpt-4o-mini",
                "context_manager": {"max_tokens": 64_000},
            },
            "agent_configs": [
                {
                    "name": "writer",
                    "system_prompt": "You write.",
                    "context_manager": {"type": "default"},  # max_tokens unset
                },
                {
                    "name": "editor",
                    "system_prompt": "You edit.",
                    "model_name": "gpt-4o",
                    "context_manager": {"max_tokens": 9_000},
                },
                {
                    "name": "archivist",
                    "system_prompt": "You keep everything.",
                    "context_manager": {"type": None},  # set: no trimming
                },
            ],
        }
    )
    team = Team(config)
    writer, editor, archivist = (
        team.agents[name] for name in ("writer", "editor", "archivist")
    )

    assert (writer.model, writer.context_manager.max_tokens) == (
        "gpt-4o-mini",
        64_000,
    )
    assert (editor.model, editor.context_manager.max_tokens) == ("gpt-4o", 9_000)
    assert (archivist.model, archivist.context_manager) == ("gpt-4o-mini", None)
    assert team._max_workers == 4  # unset, so the default


class TeamEntries(RootModel[List[TeamEntry]]):
    pass


def test_agent_entries_in_a_root_model_fall_back_to_global():
    class Config(BaseModel):
        global_: Dict[str, Any] = Field(default_factory=dict, alias="global")
        agent_configs: TeamEntries

    config = Config.model_validate(
        {"global": GLOBAL, "agent_configs": [{"name": "echo", "system_prompt": "Hi."}]}
    )
    echo = Echo(config).agents["echo"]
    assert (echo.model, echo.max_llm_recall) == ("gpt-4o-mini", 2)


class ByName(BaseModel):
    tactic_type: ClassVar[str] = "fix_echo"
    global_: Dict[str, Any] = Field(default_factory=dict, alias="global")
    agent_configs: List[AgentEntry]

    @property
    def max_workers(self) -> int:
        return 2


def test_class_vars_and_properties_of_a_config_are_read():
    from lllm.runtimes.native import Runtime
    from lllm.runtimes.native.core.tactic_registry import (
        build_tactic,
        register_tactic_class,
    )

    runtime = Runtime()
    register_tactic_class(Echo, runtime=runtime)
    config = ByName.model_validate(
        {"global": GLOBAL, "agent_configs": [{"name": "echo", "system_prompt": "Hi."}]}
    )
    tactic = build_tactic(config, runtime=runtime)  # by its tactic_type

    assert isinstance(tactic, Echo)
    assert tactic._max_workers == 2


# -- 5: ``format=`` validates the reply against the model --------------------


class Triage(BaseModel):
    team: Literal["billing", "bugs", "accounts"]
    priority: int


TRIAGE = Prompt(path="fix/triage", prompt="Route: {ticket}", format=Triage)
NOT_A_TEAM = "team: Input should be 'billing', 'bugs' or 'accounts'"


def test_format_repairs_replies_whose_fields_do_not_fit():
    agent = Tactic.quick(system_prompt="You route tickets.", model="gpt-4o-mini")
    agent.open("triage")
    agent.receive_prompt(TRIAGE, {"ticket": "Wrong company name on my invoice."})
    replies = ['{"team": "invoicing", "priority": 3}', '{"team": "billing", "priority": 3}']
    with ScriptedModel(replies) as model:
        session = agent.respond(return_session=True)

    assert session.exception_retries_count == 1
    (error,) = session.invoke_traces[0][0].execution_errors
    assert isinstance(error, ParseError)
    assert str(error) == f"The reply does not match Triage: {NOT_A_TEAM}"
    assert model.requests[1].text == (
        f"Error: The reply does not match Triage: {NOT_A_TEAM}. Please fix."
    )
    assert session.delivery.parsed == {"team": "billing", "priority": 3}


def test_format_lists_every_problem_with_its_location():
    with pytest.raises(ParseError) as info:
        TRIAGE.parse('{"team": "sales"}')
    assert str(info.value) == (
        f"The reply does not match Triage: {NOT_A_TEAM}; priority: Field required"
    )

    with pytest.raises(ParseError, match=r"Triage: Input should be an object$"):
        TRIAGE.parse("[1, 2]")
    # Invalid JSON: a ParseError with the decoder's message, as before.
    with pytest.raises(ParseError) as info:
        TRIAGE.parse("billing")
    assert str(info.value) == "Expecting value: line 1 column 1 (char 0)"
    assert isinstance(info.value.__cause__, json.JSONDecodeError)


def test_format_with_a_json_schema_needs_a_json_object():
    ask = Prompt(path="fix/schema", prompt="Answer.", format={"type": "object"})

    assert ask.parse('{"answer": 42}') == {"answer": 42}
    with pytest.raises(ParseError) as info:
        ask.parse("[42]")
    assert str(info.value) == "The reply must be a JSON object"  # no "."


# -- 6: ``parsed`` holds exactly what ``format=`` or the parser produced ----


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    amount: Decimal
    note: str = Field(default="", exclude=True)
    paid_by: str = Field(alias="paidBy")
    reference: str = Field(validation_alias=AliasChoices("reference", "ref"))

    @computed_field
    @property
    def label(self) -> str:
        return f"{self.vendor} {self.amount}"


def test_parsed_is_the_reply_json_and_validates_back_into_the_model():
    ask = Prompt(path="fix/invoice", prompt="Extract: {text}", format=Invoice)
    reply = (
        '{"vendor": "ACME", "amount": 1250.10, "note": "rush",'
        ' "paidBy": "card", "ref": "INV-7"}'
    )
    parsed = ask.parse(reply)

    assert parsed == json.loads(reply)  # a number stays a number
    invoice = Invoice.model_validate(parsed)  # no computed "label" in it
    assert (invoice.amount, invoice.note, invoice.label) == (
        Decimal("1250.1"),
        "rush",
        "ACME 1250.1",
    )


def _parse(model: Any, reply: str) -> Dict[str, Any]:
    return Prompt(path="fix/format", prompt="Answer.", format=model).parse(reply)


def test_parsed_is_the_whole_reply_object_without_defaults():
    class Plain(BaseModel):
        team: str
        urgent: bool = False
        is_global: bool = Field(default=False, alias="global")

    reply = '{"team": "bugs", "global": true, "confidence": 0.9}'
    parsed = _parse(Plain, reply)

    assert parsed == json.loads(reply)  # unfiltered; no defaults added
    assert Plain.model_validate(parsed) == Plain(team="bugs", **{"global": True})


class Envelope(BaseModel):
    team: str

    @model_validator(mode="before")
    @classmethod
    def unwrap(cls, data: Any) -> Any:
        return data.get("data", data) if isinstance(data, dict) else data


class Renamed(BaseModel):
    team: str

    @model_validator(mode="wrap")
    @classmethod
    def rename(cls, data: Any, handler: Any) -> Any:
        if isinstance(data, dict) and "group" in data:
            data = {"team": data["group"]}
        return handler(data)


class Located(BaseModel):
    name: str = Field(validation_alias=AliasPath("data", "name"))


@pytest.mark.parametrize(
    ("model", "reply"),
    [
        (RootModel[Dict[str, int]], '{"alice": 3, "bob": 5}'),
        (Envelope, '{"data": {"team": "bugs"}}'),
        (Renamed, '{"group": "bugs"}'),
        (Located, '{"data": {"name": "Ada"}}'),
    ],
    ids=["root_model", "before_validator", "wrap_validator", "alias_path"],
)
def test_parsed_validates_back_however_the_model_reads_it(model, reply):
    parsed = _parse(model, reply)

    assert parsed == json.loads(reply)
    assert model.model_validate(parsed) == model.model_validate_json(reply)


class StrictTriage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team: str
    priority: int


def as_strict_triage(text: str, **kwargs) -> dict:
    return StrictTriage.model_validate_json(text).model_dump()


class StrictRoute(Tactic):
    name = "fix_strict_route"
    agent_group = ["router"]
    output_model = StrictTriage

    def call(self, ticket: str) -> StrictTriage:
        router = self.agents["router"]
        router.open("route")
        router.receive_prompt(self.prompt, {"ticket": ticket})
        return router.respond().parsed


@pytest.mark.parametrize(
    "contract",
    [{"format": StrictTriage}, {"parser": as_strict_triage}],
    ids=["format", "parser"],
)
def test_parsed_validates_into_a_model_that_forbids_extra_fields(contract):
    route = StrictRoute(
        {
            "global": {"model_name": "gpt-4o-mini"},
            "agent_configs": [{"name": "router", "system_prompt": "You route."}],
        }
    )
    route.prompt = Prompt(path="fix/strict", prompt="Route: {ticket}", **contract)
    with ScriptedModel(['{"team": "billing", "priority": 2}']):
        result = route("I was charged twice.")

    assert result == StrictTriage(team="billing", priority=2)


def test_parser_results_are_used_as_returned():
    def percent(text: str, **kwargs) -> float:
        return float(text.rstrip("%"))

    def fields(text: str, **kwargs) -> dict:
        return {"label": text.strip()}

    assert Prompt(path="fix/percent", prompt="?", parser=percent).parse("37%") == {
        "value": 37.0
    }
    assert Prompt(path="fix/fields", prompt="?", parser=fields).parse(" bug ") == {
        "label": "bug"
    }
    assert Prompt(path="fix/plain", prompt="?").parse("hello") == {}


# -- 7: images keep their media type; overview() shows a placeholder --------


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


PNG = b"\x89PNG\r\n\x1a\n" + bytes(2040)
IMAGES = {
    "image/png": PNG,
    "image/jpeg": b"\xff\xd8\xff\xe0" + bytes(60),
    "image/webp": b"RIFF\x24\x00\x00\x00WEBPVP8 " + bytes(60),
    "image/gif": b"GIF89a" + bytes(60),
}


@pytest.mark.parametrize("media_type", sorted(IMAGES))
def test_images_are_sent_with_their_media_type(media_type):
    from lllm.runtimes.native.invokers.litellm import LiteLLMInvoker

    dialog = Dialog(owner="agent")
    dialog.put_image(_b64(IMAGES[media_type]), caption="A picture")
    (message,) = LiteLLMInvoker({})._convert_dialog(dialog)

    caption, image = message["content"]
    assert caption == {"type": "text", "text": "A picture"}
    url = image["image_url"]["url"]
    assert url == f"data:{media_type};base64,{_b64(IMAGES[media_type])}"


def test_put_image_rejects_riff_data_that_is_not_webp():
    wav = b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(60)
    with pytest.raises(ValueError, match="PNG/JPEG/WebP/GIF"):
        Dialog(owner="agent").put_image(_b64(wav))


def test_overview_shows_an_image_as_type_size_and_caption():
    dialog = _dialog()
    dialog.put_image(_b64(PNG), caption="Revenue by quarter")
    dialog.put_image(_b64(IMAGES["image/gif"]))
    dialog.put_text("What is the trend?")

    assert dialog.overview().split("\n\n") == [
        "[0. system (developer)]: You help.",
        "[1. user (user)]: <image/png, 2.0 KB> Revenue by quarter",
        "[2. user (user)]: <image/gif, 66 B>",
        "[3. user (user)]: What is the trend?",
    ]


# -- 8: ``extra_settings`` from the config reach the agent -------------------


class Styled(Tactic):
    name = "fix_styled"
    agent_group = ["writer"]

    def call(self, key: str) -> Any:
        settings = self.agents["writer"].extra_settings
        settings["calls"] = settings.get("calls", 0) + 1  # per call
        return settings[key]


def test_extra_settings_reach_the_agent():
    styled = Styled(
        {
            "global": {
                "model_name": "gpt-4o-mini",
                "extra_settings": {"style": "terse", "audience": "developers"},
            },
            "agent_configs": [
                {
                    "name": "writer",
                    "system_prompt": "You write.",
                    "extra_settings": {"style": "formal"},
                }
            ],
        }
    )

    assert styled.agents["writer"].extra_settings == {
        "style": "formal",
        "audience": "developers",
    }
    assert styled("style") == "formal"
    assert styled("calls") == 1  # every call's agent gets its own copy
    assert styled("calls") == 1
    assert Tactic.quick(model="gpt-4o-mini").extra_settings == {}


WRITER = {"model_name": "gpt-4o-mini", "system_prompt": "You write."}


@pytest.mark.parametrize("value", [[], "", 0, False, ["style"]])
def test_extra_settings_must_be_a_mapping(value):
    from lllm.runtimes.native import AgentSpec

    with pytest.raises(TypeError, match="'extra_settings' must be a mapping"):
        AgentSpec.from_config("writer", {**WRITER, "extra_settings": value})


def test_extra_settings_take_any_mapping_and_values_deepcopy_cannot_copy():
    import threading
    from types import MappingProxyType

    from lllm.runtimes.native import AgentSpec, Runtime

    lock = threading.Lock()
    settings = MappingProxyType({"lock": lock, "style": "terse"})
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": settings})
    first, second = (spec.build(Runtime(), object()) for _ in range(2))

    assert first.extra_settings == {"lock": lock, "style": "terse"}
    assert first.extra_settings["lock"] is lock  # shared, not copied
    assert first.extra_settings is not second.extra_settings
    unset = AgentSpec.from_config("writer", {**WRITER, "extra_settings": None})
    assert unset.extra_settings == {}


class NoCopies:
    """A live object, such as a client, that must never be deep-copied."""

    def __deepcopy__(self, memo):
        raise AssertionError("deep-copied a live object")


def test_extra_settings_copy_plain_data_and_share_other_objects():
    from lllm.runtimes.native import AgentSpec, Runtime

    client = NoCopies()
    settings = {"client": client, "tags": ["a"], "limits": {"words": [100]}}
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": settings})
    first, second = (spec.build(Runtime(), object()) for _ in range(2))

    first.extra_settings["tags"].append("b")
    first.extra_settings["limits"]["words"].append(200)
    settings["tags"].append("c")  # the config given, after parsing
    for owner in (second, spec):
        assert getattr(owner, "extra_settings") == {
            "client": client,
            "tags": ["a"],
            "limits": {"words": [100]},
        }
    assert first.extra_settings["client"] is client


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_an_sdk_client_in_extra_settings_is_shared_cleanly():
    import gc

    openai = pytest.importorskip("openai")
    from lllm.runtimes.native import AgentSpec, Runtime

    client = openai.OpenAI(api_key="sk-test", base_url="http://localhost:9")
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": {"client": client}})
    agents = [spec.build(Runtime(), object()) for _ in range(2)]
    gc.collect()  # a half-copied client would complain here

    assert all(agent.extra_settings["client"] is client for agent in agents)


def test_extra_settings_copies_keep_cycles_sharing_and_container_types():
    from collections import Counter, defaultdict

    from lllm.runtimes.native import AgentSpec, Runtime

    Point = namedtuple("Point", "x y")
    loop = ["a"]
    loop.append(loop)  # a list that contains itself
    shared = [1, 2]
    settings = {
        "loop": loop,
        "first": shared,
        "second": shared,
        "counts": Counter(a=2),
        "groups": defaultdict(list, {"k": [1]}),
        "point": Point(x=[1], y=2),
    }
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": settings})
    copied = spec.build(Runtime(), object()).extra_settings

    assert copied["loop"][1] is copied["loop"] is not loop
    assert copied["first"] is copied["second"] is not shared  # still shared
    counts, groups, point = copied["counts"], copied["groups"], copied["point"]
    assert type(counts) is Counter and counts == Counter(a=2)
    assert counts is not settings["counts"]
    assert type(groups) is defaultdict and groups.default_factory is list
    assert groups == {"k": [1]} and groups["k"] is not settings["groups"]["k"]
    assert type(point) is Point and point == Point(x=[1], y=2)
    assert point.x is not settings["point"].x


def test_extra_settings_cycles_through_tuples_and_at_the_top_close_on_the_copy():
    import time

    from lllm.runtimes.native import AgentSpec, Runtime

    history: list = []
    entry = ("start", history)
    history.append(entry)  # a cycle through a tuple
    when = time.gmtime(0)  # a struct_time: hidden fields, not rebuilt
    # The tuple comes first, so the copy meets the cycle through it.
    settings: Dict[str, Any] = {"entry": entry, "history": history, "when": when}
    settings["self"] = settings  # a cycle at the top
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": settings})
    copied = spec.build(Runtime(), object()).extra_settings

    assert copied["self"] is copied
    assert copied["history"] is not history
    assert copied["history"][0] is copied["entry"]  # one copy of the tuple
    assert copied["entry"][1] is copied["history"]  # closes on the copy
    assert copied["when"] is when
    assert not _reaches(copied, history) and not _reaches(copied, entry)


def _reaches(root: Any, target: Any) -> bool:
    """Whether ``target`` itself can be reached from ``root``."""
    seen, stack = set(), [root]
    while stack:
        value = stack.pop()
        if value is target:
            return True
        if id(value) not in seen:
            seen.add(id(value))
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
    return False


def _nested(levels: int, wrap: Any) -> Any:
    value: Any = ["end"]
    for _ in range(levels):
        value = wrap(value)
    return value


@pytest.mark.parametrize(
    "wrap",
    [lambda v: [v], lambda v: (v,), lambda v: {0: v}],
    ids=["lists", "tuples", "dicts"],
)
def test_extra_settings_too_deep_to_copy_raise_a_clear_error(wrap):
    from lllm.runtimes.native import AgentSpec

    message = "Agent 'writer': 'extra_settings' is nested more than 500 levels deep"
    with pytest.raises(ValueError, match=message):  # the same on every Python
        deep = _nested(2_000, wrap)
        AgentSpec.from_config("writer", {**WRITER, "extra_settings": {"d": deep}})
    # Just under the bound it copies, down to the end.
    shallow = _nested(490, wrap)
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": {"d": shallow}})
    original, copied = shallow, spec.extra_settings["d"]
    for _ in range(490):
        original, copied = original[0], copied[0]
    assert copied == ["end"] and copied is not original


def _levels_that_fit() -> int:
    """How many nested class instantiations fit on the stack from here.

    Class calls are how a deep caller stack hides depth on Python 3.10 and
    3.11, where the C-level call counts toward the limit without a frame.
    """
    reached = [0]

    class Descend:
        def __init__(self) -> None:
            reached[0] += 1
            try:
                Descend()
            except RecursionError:
                pass

    Descend()
    return reached[0]


def test_settings_copy_fully_under_a_deep_caller_stack():
    from collections import OrderedDict

    from lllm.runtimes.native import AgentSpec

    def mixed(levels: int) -> Any:  # lists and OrderedDicts, alternating
        value: Any = ["end"]
        for level in range(levels):
            value = OrderedDict(v=value) if level % 2 else [value]
        return value

    class Deep:
        def __init__(self, more: int, out: list) -> None:
            if more:
                Deep(more - 1, out)
            else:
                settings = {"d": mixed(400)}
                config = {**WRITER, "extra_settings": settings}
                out.append((settings, AgentSpec.from_config("writer", config)))

    out: list = []
    # 150 levels short of the limit (the estimate can be ~70 levels high on
    # CPython 3.14): plenty for the copy, far too few for a recursive copy
    # of the 400 levels below, within the 500-level policy.
    Deep(_levels_that_fit() - 150, out)
    (settings, spec), = out

    original, copied = settings["d"], spec.extra_settings["d"]
    for level in reversed(range(400)):  # nothing below is shared
        assert copied is not original and type(copied) is type(original)
        index = "v" if level % 2 else 0
        original, copied = original[index], copied[index]
    assert copied == ["end"] and copied is not original


class Short(namedtuple("Short", "a b")):
    """A namedtuple whose iteration yields fewer items than its length."""

    def __iter__(self):
        return iter(tuple.__getitem__(self, slice(0, 1)))


def test_a_namedtuple_that_iterates_short_fails_rather_than_pads():
    from lllm.runtimes.native import AgentSpec

    with pytest.raises(TypeError):  # as copying it always did; never padded
        AgentSpec.from_config("writer", {**WRITER, "extra_settings": {"s": Short(1, 2)}})


class Meddler(dict):
    """Copying it grows the dict, set, or list that holds it."""

    def __init__(self, holder: Any) -> None:
        super().__init__(x=1)
        self.holder = holder

    def __hash__(self) -> int:  # so that a set can hold it
        return id(self)

    def __copy__(self):
        holder = self.holder  # still being read
        if isinstance(holder, dict):
            holder["late"] = 0
        elif isinstance(holder, set):
            holder.add("late")
        else:
            holder.append("late")
        return Meddler(holder)


@pytest.mark.parametrize(
    "kind, error, match",
    [
        (dict, RuntimeError, "changed size during iteration"),
        (set, RuntimeError, "changed size during iteration"),
        # A list is read by index, as ever: grown, it overruns its copy.
        (list, IndexError, "out of range"),
    ],
)
def test_a_container_changed_during_the_copy_is_not_ignored(kind, error, match):
    from lllm.runtimes.native import AgentSpec

    holder = kind()
    meddler = Meddler(holder)
    if kind is dict:
        holder["m"] = meddler
    elif kind is set:
        holder.add(meddler)
    else:
        holder.append(meddler)
    with pytest.raises(error, match=match):
        AgentSpec.from_config("writer", {**WRITER, "extra_settings": {"h": holder}})


def test_copying_a_wide_list_takes_about_its_own_size_in_memory():
    import tracemalloc

    from lllm.runtimes.native import AgentSpec

    AgentSpec.from_config("writer", {**WRITER, "extra_settings": {"w": [1]}})
    settings = {"w": list(range(200_000))}  # its copy takes 1.6 MB
    ours = not tracemalloc.is_tracing()  # else leave the user's tracing on
    if ours:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before, _ = tracemalloc.get_traced_memory()
        AgentSpec.from_config("writer", {**WRITER, "extra_settings": settings})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if ours:
            tracemalloc.stop()

    # Listing the children up front, an entry each, peaked at 36 MiB.
    assert peak - before < 4 * 2**20


class Stopwords(set):
    """Seeds defaults when it is made from nothing."""

    def __init__(self, words=()):
        super().__init__(words or {"a", "an", "the"})


class QuietStopwords(Stopwords):
    """Says it is empty, defaults or not."""

    def __bool__(self) -> bool:
        return False


class TouchyStopwords(Stopwords):
    """Cannot be asked whether it is empty."""

    def __bool__(self) -> bool:
        raise TypeError("ambiguous")


class GivenStopwords(Stopwords):
    """Counts only the words it was given, not the defaults it seeded."""

    def __init__(self, words=()):
        super().__init__(words)
        self.given = len(set(words))

    def __len__(self) -> int:
        return self.given


@pytest.mark.parametrize(
    "kind", [Stopwords, QuietStopwords, TouchyStopwords, GivenStopwords]
)
def test_an_empty_set_whose_copy_seeds_defaults_comes_back_empty(kind):
    from lllm.runtimes.native import AgentSpec, Runtime

    empty = kind()
    empty.clear()  # the user's set is empty
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": {"s": empty}})
    copied = spec.build(Runtime(), object()).extra_settings["s"]

    assert type(copied) is kind and copied == set()


class CopiesToDict(set):
    """Its copy is a dict."""

    def __copy__(self):
        return {}


@pytest.mark.parametrize(
    "make, kind",
    [(CopiesToDict, dict), (lambda: MagicMock(spec=set), MagicMock)],
    ids=["copies to a dict", "spec mock"],
)
def test_a_set_whose_copy_is_not_a_real_set_is_copied_as_ever(make, kind):
    from lllm.runtimes.native import AgentSpec

    original = make()
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": {"s": original}})
    copied = spec.extra_settings["s"]

    assert isinstance(copied, kind) and copied is not original  # as 68ef9fd


class CopiesFail(dict):
    """A container that cannot be copied, counting the attempts."""

    attempts = 0

    def __copy__(self):
        CopiesFail.attempts += 1
        raise TypeError("cannot be copied")


def test_a_container_that_cannot_be_copied_is_tried_once_per_copy():
    from lllm.runtimes.native import AgentSpec, Runtime

    shared = CopiesFail(a=1)
    settings = {"x": shared, "y": shared, "z": [shared]}
    CopiesFail.attempts = 0
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": settings})
    assert CopiesFail.attempts == 1
    copied = spec.build(Runtime(), object()).extra_settings
    assert CopiesFail.attempts == 2  # once more, for the agent's own copy
    assert copied["x"] is shared and copied["z"][0] is shared


class Settings(dict):
    """Reads missing attributes from a parent: copying it recurses."""

    def __init__(self, *args: Any, parent: Any = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.parent = parent

    def __getattr__(self, name: str) -> Any:
        if self.parent is not None:  # loops on a copy not yet initialized
            return getattr(self.parent, name)
        raise AttributeError(name)


def test_a_container_whose_copy_recurses_is_shared():
    from lllm.runtimes.native import AgentSpec, Runtime

    settings = Settings(tags=["a"], parent=Settings())
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": {"s": settings}})

    assert spec.build(Runtime(), object()).extra_settings["s"] is settings


class Immutable(dict):
    """Like werkzeug's ImmutableDict: copying it gives it back."""

    def __copy__(self):
        return self

    def __setitem__(self, key, value):
        raise TypeError("immutable")


class FrozenCopies(dict):
    """Its copies are frozen, like a frozen python-box Box."""

    frozen = False

    def __copy__(self):
        twin = FrozenCopies(self)
        twin.frozen = True
        return twin

    def __setitem__(self, key, value):
        if self.frozen:
            raise TypeError("frozen")
        super().__setitem__(key, value)


class FrozenListCopies(list):
    """A list whose copies are frozen."""

    frozen = False

    def __copy__(self):
        twin = FrozenListCopies(self)
        twin.frozen = True
        return twin

    def __setitem__(self, index, value):
        if self.frozen:
            raise TypeError("frozen")
        super().__setitem__(index, value)


class FrozenSetCopies(set):
    """A set whose copies are frozen (clear and update still work)."""

    frozen = False

    def __copy__(self):
        twin = FrozenSetCopies(self)
        twin.frozen = True
        return twin

    def add(self, item):
        if self.frozen:
            raise TypeError("frozen")
        super().add(item)

    def discard(self, item):
        if self.frozen:
            raise TypeError("frozen")
        super().discard(item)


def test_read_only_lists_and_sets_are_shared_too():
    from lllm.runtimes.native import AgentSpec, Runtime

    frozen_list, frozen_set = FrozenListCopies([["a"]]), FrozenSetCopies({"b"})
    settings = {"list": frozen_list, "set": frozen_set}
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": settings})
    copied = spec.build(Runtime(), object()).extra_settings

    assert copied["list"] is frozen_list and copied["set"] is frozen_set


class ReadOnlySet(set):
    """A set whose copies reject every change."""

    frozen = False

    def __copy__(self):
        twin = ReadOnlySet(self)
        twin.frozen = True
        return twin

    def _check(self) -> None:
        if self.frozen:
            raise TypeError("read-only")

    def add(self, item):
        self._check()
        super().add(item)

    def discard(self, item):
        self._check()
        super().discard(item)

    def clear(self):
        self._check()
        super().clear()

    def update(self, *others):
        self._check()
        super().update(*others)


def test_an_empty_read_only_set_is_copied_without_a_fill_in():
    from lllm.runtimes.native import AgentSpec, Runtime

    empty = ReadOnlySet()  # nothing to probe, and nothing to fill in
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": {"e": empty}})
    copied = spec.build(Runtime(), object()).extra_settings["e"]

    assert type(copied) is ReadOnlySet and copied == set() and copied is not empty


class Tags(list):
    pass


class Labels(set):
    pass


def test_writable_list_and_set_subclasses_are_copied_per_agent():
    from lllm.runtimes.native import AgentSpec, Runtime

    tags, labels = Tags(["a"]), Labels({"b"})
    settings = {"tags": tags, "labels": labels}
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": settings})
    first, second = (spec.build(Runtime(), object()).extra_settings for _ in range(2))

    assert type(first["tags"]) is Tags and type(first["labels"]) is Labels
    first["tags"].append("x")
    first["labels"].add("y")
    for owner in (second, spec.extra_settings, settings):
        assert owner["tags"] == ["a"] and owner["labels"] == {"b"}


class SelfCopies(dict):
    """Its copy is itself, yet it takes writes."""

    def __copy__(self):
        return self


def test_read_only_containers_in_extra_settings_are_shared():
    from lllm.runtimes.native import AgentSpec, Runtime

    tags = ["c"]
    immutable, frozen = Immutable(tags=["a"]), FrozenCopies(tags=["b"])
    shared = SelfCopies(tags=tags)
    settings = {
        "immutable": immutable,
        "frozen": frozen,
        "again": frozen,
        "shared": shared,
    }
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": settings})
    copied = spec.build(Runtime(), object()).extra_settings

    assert copied["immutable"] is immutable
    assert copied["frozen"] is frozen and copied["again"] is frozen
    # Shared, and not filled in: the user's object keeps its own list.
    assert copied["shared"] is shared and shared["tags"] is tags


def test_a_read_only_container_is_shared_before_its_children_are_copied():
    from lllm.runtimes.native import AgentSpec, Runtime

    loop: list = []
    frozen = FrozenCopies()
    dict.__setitem__(frozen, "loop", loop)
    loop.append(frozen)  # a cycle through the read-only container
    settings = {"frozen": frozen, "loop": loop}
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": settings})
    copied = spec.build(Runtime(), object()).extra_settings

    assert copied["frozen"] is frozen
    assert copied["loop"][0] is frozen  # no half-built copy of it


class Unlistable(dict):
    """Copies, and takes writes, but its items cannot be listed."""

    def __copy__(self):
        return Unlistable(dict.items(self))

    def items(self):
        raise RuntimeError("cannot list its items")


@pytest.mark.parametrize("depth", [1, 2])
def test_a_container_whose_items_cannot_be_listed_fails_alike_at_any_depth(depth):
    from lllm.runtimes.native import AgentSpec

    value: Any = Unlistable(a=1)
    for _ in range(depth - 1):
        value = {"inside": value}  # a writable parent, not shared instead
    with pytest.raises(RuntimeError, match="cannot list its items"):
        AgentSpec.from_config("writer", {**WRITER, "extra_settings": {"v": value}})


def _released(log: List[str], name: str, items: Any) -> Any:
    """Yield ``items``, and log ``name`` once released, used up or not."""
    try:
        yield from items
    finally:
        log.append(name)


class Streamed(dict):
    """Lists its items from a generator that holds a resource until it is
    released; its copies refuse the key "refused"."""

    def __init__(self, log: List[str], name: str, **items: Any) -> None:
        super().__init__(**items)
        self.log, self.name = log, name

    def __copy__(self):
        return type(self)(self.log, self.name, **self)

    def __setitem__(self, key, value):
        if key == "refused":
            raise TypeError("refused")
        super().__setitem__(key, value)

    def items(self):
        return _released(self.log, self.name, dict.items(self))


class Zipped(Streamed):
    """Lists its items through zip, which stops when the keys run out: the
    generator of values stays open, though zip is used up."""

    def items(self):
        values = _released(self.log, self.name, dict.values(self))
        return zip(dict.keys(self), values)


class Unbuildable(namedtuple("Unbuildable", "a b")):
    """Reads its items through zip, as Zipped does, and cannot be rebuilt."""

    def __new__(cls, log: List[str], a: Any, b: Any):
        pair = super().__new__(cls, a, b)
        pair.log = log
        return pair

    def __iter__(self):
        values = _released(self.log, "pair", tuple.__iter__(self))
        return map(itemgetter(1), zip(range(2), values))

    @classmethod
    def _make(cls, iterable):
        raise TypeError("cannot be rebuilt")


class Interrupted(dict):
    """Its copy is cut short, as by Ctrl-C."""

    def __copy__(self):
        raise KeyboardInterrupt


def _three_deep(log: List[str]) -> Any:
    inner = Streamed(log, "inner", a=Short(1, 2))
    return Streamed(log, "outer", m=Streamed(log, "middle", x=[inner]))


def _refused_when_done(log: List[str]) -> Any:
    # "a" first, so that the read-only probe passes and the copy goes on.
    return Streamed(log, "parent", a=1, refused=Zipped(log, "zipped", a=1))


#: Each case's settings value, and what it releases, in order.
RELEASES = {
    # A child fails to rebuild, cannot be listed, is too deep, is refused
    # by the copy, or has its copy interrupted (a BaseException).
    "rebuild": (lambda log: Streamed(log, "s", a=Short(1, 2)), ["s"]),
    "listing": (lambda log: Streamed(log, "s", a=Unlistable(a=1)), ["s"]),
    "depth": (lambda log: Streamed(log, "s", a=_nested(600, lambda v: [v])), ["s"]),
    "storing": (lambda log: Streamed(log, "s", a=1, refused=2), ["s"]),
    "interrupt": (lambda log: Streamed(log, "s", a=Interrupted()), ["s"]),
    "nested": (_three_deep, ["inner", "middle", "outer"]),
    # A used-up iterator can hold a generator open too: a finished child
    # its parent refuses, and a tuple whose rebuild fails.
    "used-up child": (_refused_when_done, ["zipped", "parent"]),
    "used-up tuple": (lambda log: Unbuildable(log, 1, 2), ["pair"]),
}


@pytest.mark.parametrize("case", RELEASES)
def test_a_failed_copy_releases_its_iterators_before_the_caller_sees_it(case):
    from lllm.runtimes.native import AgentSpec

    build, released = RELEASES[case]
    log: List[str] = []
    value = build(log)
    try:
        AgentSpec.from_config("writer", {**WRITER, "extra_settings": {"v": value}})
    except (TypeError, RuntimeError, ValueError, KeyboardInterrupt):
        log.append("caught")

    # Innermost first, as a recursive copy's unwinding releases them, and
    # not once the caller drops the error.
    assert log == [*released, "caught"]


def test_values_built_on_access_each_get_their_own_copy(tmp_path):
    import shelve

    from lllm.runtimes.native import AgentSpec

    expected = {f"k{n}": {"tags": [n], "pair": (n, [n])} for n in range(200)}
    # A shelf unpickles each value as it is read, and frees it after: the
    # copy must keep originals alive, or a new one reuses an old address.
    with shelve.open(str(tmp_path / "settings")) as shelf:
        shelf.update(expected)
        spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": shelf})

    assert spec.extra_settings == expected


def test_a_spec_made_directly_takes_any_extra_settings():
    import dataclasses

    from lllm.runtimes.native import AgentSpec, Runtime

    spec = AgentSpec.from_config("writer", WRITER)
    unset = dataclasses.replace(spec, extra_settings=None)
    assert unset.build(Runtime(), object()).extra_settings == {}

    listed = dataclasses.replace(spec, extra_settings=["a", ["b"]])
    settings = listed.build(Runtime(), object()).extra_settings
    assert settings == ["a", ["b"]] and settings[1] is not listed.extra_settings[1]


# -- a failed agent call keeps its record ------------------------------------


RATE = Prompt(path="fix/rate-reply", prompt="Rate 1-5: {review}", format=Rating)
NOT_JSON = ["Four stars.", "Four stars, really."]


def _replies(call: Any) -> list:
    return [t.message.content for ts in call.invoke_traces.values() for t in ts]


def test_a_failed_respond_carries_its_session():
    from lllm.runtimes.native import AgentException

    agent = Tactic.quick(
        system_prompt="You rate reviews.", model="gpt-4o-mini", max_exception_retry=1
    )
    agent.open("rate")
    agent.receive_prompt(RATE, {"review": "Fast shipping, crushed box."})
    with ScriptedModel(NOT_JSON), pytest.raises(AgentException) as info:
        agent.respond()

    call = info.value.agent_session
    assert (call.state, call.exception_retries_count) == ("failure", 1)
    assert _replies(call) == NOT_JSON


class Rate(Tactic):
    name = "fix_rate"
    agent_group = ["rater"]

    def call(self, review: str) -> dict:
        rater = self.agents["rater"]
        rater.open("rate")
        rater.receive_prompt(RATE, {"review": review})
        return rater.respond().parsed


def test_a_tactic_session_records_the_agent_call_that_failed():
    from lllm.runtimes.native import AgentException

    sessions = []

    class Keep:
        def on_tactic_failure(self, *, session, **_):
            sessions.append(session)

    rate = Rate(
        {
            "global": {"model_name": "gpt-4o-mini", "max_exception_retry": 1},
            "agent_configs": [{"name": "rater", "system_prompt": "You rate."}],
        },
        observer=Keep(),
    )
    with ScriptedModel(NOT_JSON), pytest.raises(AgentException):
        rate("Fast shipping, crushed box.")

    (session,) = sessions
    (call,) = session.agent_sessions["rater"]
    assert (call.state, _replies(call)) == ("failure", NOT_JSON)


# -- 9: context managers and stream handlers are exported -------------------


def test_extension_base_classes_are_exported_from_the_native_runtime():
    import lllm.runtimes.native as native
    from lllm.runtimes.native.core import dialog
    from lllm.runtimes.native.invokers import base

    exported = {
        "BaseInvoker": base.BaseInvoker,
        "BaseStreamHandler": base.BaseStreamHandler,
        "ContextManager": dialog.ContextManager,
        "DefaultContextManager": dialog.DefaultContextManager,
    }
    for name, value in exported.items():
        assert getattr(native, name) is value
        assert name in native.__all__


# -- 10: str(InvokeCost) shows rates per million tokens ----------------------


@pytest.mark.parametrize(
    ("input_rate", "output_rate", "shown"),
    [
        (0.15e-6, 0.6e-6, "In: $0.15/1M tok, Out: $0.60/1M tok"),  # gpt-4o-mini
        (0.0375e-6, 0.15e-6, "In: $0.0375/1M tok, Out: $0.15/1M tok"),
        (15e-6, 75e-6, "In: $15.00/1M tok, Out: $75.00/1M tok"),
    ],
)
def test_invoke_cost_shows_rates_per_million_tokens(input_rate, output_rate, shown):
    cost = InvokeCost(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        input_cost_per_token=input_rate,
        output_cost_per_token=output_rate,
    )
    assert f" | Rates: [{shown}]\n" in str(cost)


def test_reply_cost_shows_the_model_rates():
    with ScriptedModel(["Hi."]):
        reply = Tactic.quick("Hello", model="gpt-4o-mini")
    assert "Rates: [In: $0.15/1M tok, Out: $0.60/1M tok]" in str(reply.cost)


# -- 11: Tactic.quick() takes agent settings as agent settings --------------


def test_quick_sets_agent_fields_and_sends_the_rest_as_model_args():
    agent = Tactic.quick(
        system_prompt="You answer briefly.",
        model="gpt-4o-mini",
        max_llm_recall=2,
        max_exception_retry=1,
        max_interrupt_steps=2,
        max_rate_limit_retries=0,
        extra_settings={"team": "docs"},
        temperature=0.1,
        model_args={"seed": 7, "temperature": 0.9},
    )

    assert agent.name == "assistant"
    assert agent.max_llm_recall == 2
    assert agent.max_exception_retry == 1
    assert agent.max_interrupt_steps == 2
    assert agent.max_rate_limit_retries == 0
    assert agent.extra_settings == {"team": "docs"}
    # A keyword wins over the same key in model_args=.
    assert agent.model_args == {"seed": 7, "temperature": 0.1}

    agent.open("chat")
    agent.receive("Hi.")
    with ScriptedModel(["Hello."]):
        session = agent.respond(return_session=True)
    assert session.max_llm_recall == 2
    assert session.invoke_traces[0][0].model_args == {
        "seed": 7,
        "temperature": 0.1,
        "drop_params": True,
        "stream": False,
    }


def test_quick_takes_a_stream_handler_and_a_name():
    from lllm.runtimes.native import BaseStreamHandler

    class Collect(BaseStreamHandler):
        def __init__(self):
            self.pieces = []

        def handle_chunk(self, chunk_content, chunk_response):
            self.pieces.append(chunk_content)

    collect = Collect()
    with ScriptedModel(["Hello there."]):
        reply = Tactic.quick(
            "Hi", model="gpt-4o-mini", stream_handler=collect, name="greeter"
        )

    assert reply.name == "greeter"
    assert "".join(collect.pieces) == reply.content == "Hello there."


# -- 12: Responses API usage counts prompt and completion tokens ------------


def test_responses_api_usage_counts_prompt_and_completion_tokens(monkeypatch):
    from types import SimpleNamespace

    from litellm.types.llms.openai import ResponseAPIUsage

    from lllm.runtimes.native import APITypes
    from lllm.runtimes.native.invokers import litellm as invoker_module

    usage = ResponseAPIUsage(
        input_tokens=1200,
        output_tokens=300,
        total_tokens=1500,
        input_tokens_details={"cached_tokens": 1000},
        output_tokens_details={"reasoning_tokens": 200},
    )
    response = SimpleNamespace(
        output=[],
        output_text="Paris.",
        usage=usage,
        reasoning=None,
        _hidden_params={"response_cost": 0.00036},
    )
    monkeypatch.setattr(invoker_module, "responses_api", lambda **kwargs: response)

    agent = Tactic.quick(
        system_prompt="Be brief.", model="gpt-4o-mini", api_type=APITypes.RESPONSE
    )
    agent.open("q")
    agent.receive("What is the capital of France?")
    cost = agent.respond().cost

    assert (cost.prompt_tokens, cost.completion_tokens, cost.total_tokens) == (
        1200,
        300,
        1500,
    )
    assert (cost.cached_prompt_tokens, cost.reasoning_tokens) == (1000, 200)
    assert cost.prompt_cost == pytest.approx(1200 * 0.15e-6)
    assert cost.completion_cost == pytest.approx(300 * 0.6e-6)
    assert cost.cost == pytest.approx(0.00036)


# -- a deeply nested task does not overflow a small thread stack -----------


DEEP_TASKS = """
import asyncio
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict

threading.stack_size(1 << 20)  # 1 MiB, as on python:alpine

from pydantic import BaseModel

from lllm import Tactic


def deep(levels):
    value = []
    for _ in range(levels):
        value = [value]
    return value


class Notes(BaseModel):
    data: Dict[str, Any]


@dataclass
class Box:
    items: Any


chain = OrderedDict()
for _ in range(9_000):
    chain = OrderedDict(next=chain)


class Kind(Tactic):
    name = "kind"

    def call(self, task):
        return type(task).__name__


tasks = [
    {"next": chain},  # a dict, as an HTTP body would be
    Notes(data={"x": deep(9_000)}),
    chain,
    Box(items=deep(9_000)),
    {"notes": Notes(data={"x": deep(9_000)})},
    # Not walked, named by their type: their repr would recurse in C.
    deque([deep(9_000)]),
    SimpleNamespace(items=deep(9_000)),
]
for task in tasks:
    print(asyncio.run(Kind().arun(task)))  # arun works on a worker thread
"""


def _run_apart(script: str, timeout: int = 120) -> Any:
    """Run a script in its own process: a crash (SIGSEGV) or a hang would
    take pytest down with it."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root)}
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def test_deeply_nested_tasks_run_on_a_small_thread_stack():
    result = _run_apart(DEEP_TASKS)

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split() == [
        "dict",
        "Notes",
        "OrderedDict",
        "Box",
        "dict",
        "deque",
        "SimpleNamespace",
    ]


QUICK_NAMES = """
import time

from pydantic import BaseModel, ConfigDict

from lllm.runtimes.native.core.tactic import _task_to_string


class Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


loop = {}
model = Loose()
for n in range(10_000):  # 10,000 ways back to itself
    loop[f"k{n}"] = loop
    model.__pydantic_extra__[f"k{n}"] = model
deep = [{n: n for n in range(10_000)}] * 5_000  # one dict, 5,000 times...
for _ in range(19):
    deep = [deep]  # ...at depth 20, where the walk stops
for task in (loop, model, deep):
    start = time.perf_counter()
    _task_to_string(task)
    print(round(time.perf_counter() - start, 3))
"""


def test_self_references_and_the_depth_bound_name_sessions_quickly():
    result = _run_apart(QUICK_NAMES, timeout=60)

    assert result.returncode == 0, result.stderr[-2000:]
    times = [float(seconds) for seconds in result.stdout.split()]
    assert len(times) == 3 and max(times) < 0.5


def test_session_names_hash_on_fips_mode_python(monkeypatch):
    import hashlib

    real_md5 = hashlib.md5

    def fips_md5(data=b"", *, usedforsecurity=True):
        if usedforsecurity:  # as FIPS-mode Python refuses md5
            raise ValueError("[digital envelope routines] unsupported")
        return real_md5(data, usedforsecurity=False)

    monkeypatch.setattr(hashlib, "md5", fips_md5)
    names = []

    class Names:
        def observe_tactic(self, *, session_name, **_):
            names.append(session_name)

    class Echoes(Tactic):
        name = "fix_fips"

        def call(self, task):
            return task

    assert Echoes(observer=Names())({"a": 1}) == {"a": 1}
    assert names[0].startswith("fix_fips_")


def test_session_names_hash_equal_tasks_alike_and_others_apart():
    from lllm.runtimes.native.core.tactic import _task_to_string as text

    task = {"ticket": "Refund", "lines": [1, 2.5, None, True, b"\x00"]}
    assert text(task) == text({**task, "lines": list(task["lines"])})  # a copy
    assert text(task) != text({**task, "lines": [1, 2.5, None, True, b"\x01"]})
    assert text({"a": 1}) != text({"a": "1"})
    assert text("plain text") == "plain text"
    # At most 10,000 values are read, so a huge list stays cheap.
    assert text(list(range(10_000)) + [0]) == text(list(range(10_000)) + [1])

    names = []

    class Names:
        def observe_tactic(self, *, session_name, **_):
            names.append(session_name)

    class Kind(Tactic):
        name = "fix_kind"

        def call(self, task):
            return type(task).__name__

    kind = Kind(observer=Names())
    for value in (task, dict(task), {"other": True}, "\ud800 a lone surrogate"):
        kind(value)
    # {tactic}_{hash8}_{YYYYmmdd}_{HHMMSS}
    prefixes = [name.rsplit("_", 3)[:2] for name in names]
    assert prefixes[0] == prefixes[1] == ["fix_kind", prefixes[0][1]]
    assert prefixes[2] != prefixes[0]
    assert len(prefixes[0][1]) == 8
