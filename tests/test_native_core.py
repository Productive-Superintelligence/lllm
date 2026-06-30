from types import MappingProxyType

import pytest
from pydantic import ValidationError

from lllm.runtimes.native import (
    DefaultTagParser,
    Dialog,
    Function,
    FunctionCall,
    Message,
    ParseError,
    Prompt,
    Role,
    TokenLogprob,
    find_md_blocks,
    find_xml_blocks,
    tool,
)


def test_prompt_renders_extends_and_reports_metadata():
    prompt = Prompt(path="draft", prompt="Write about {topic}.", metadata={"v": 1})

    assert prompt.template_vars == {"topic"}
    assert prompt(topic="runtime boundaries") == "Write about runtime boundaries."
    assert prompt.info_dict()["prompt_hash"]

    child = prompt.extend(path="draft/brief", prompt="Briefly: {topic}")

    assert child.path == "draft/brief"
    assert child(topic="tests") == "Briefly: tests"


def test_tool_schema_and_execution_preserve_function_call_result():
    @tool(description="Add two values", prop_desc={"left": "Left side"})
    def add(left: int, right: int = 1) -> int:
        return left + right

    assert add.properties["left"]["type"] == "integer"
    assert add.required == ["left"]

    call = add(FunctionCall(name="add", arguments={"left": 2, "right": 3}))

    assert call.success
    assert call.result == 5
    assert "Return of calling function add" in call.result_str
    assert add.to_tool()["function"]["parameters"]["required"] == ["left"]


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        ".",
        "..",
        "bad/name",
        "bad:name",
        "bad\\name",
        "bad%2Fname",
    ],
)
def test_native_function_names_reject_malformed_tokens(name):
    with pytest.raises(ValidationError):
        Function(name=name, description="Bad name.", properties={})
    with pytest.raises(ValidationError):
        FunctionCall(name=name)

    def helper(value: str) -> str:
        return value

    with pytest.raises(ValidationError):
        Function.from_callable(helper, name=name)
    with pytest.raises(ValidationError):
        tool(name=name)(helper)


@pytest.mark.parametrize("prop_desc", [[], [("value", "desc")], "bad", 123])
def test_native_function_property_descriptions_reject_non_mapping_inputs(prop_desc):
    def helper(value: str) -> str:
        return value

    with pytest.raises(TypeError, match="prop_desc"):
        Function.from_callable(helper, prop_desc=prop_desc)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="prop_desc"):
        tool(prop_desc=prop_desc)(helper)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "prop_desc",
    [
        {b"value": "Description."},
        {"value": b"Description."},
        {1: "Description."},
        {"value": object()},
    ],
)
def test_native_function_property_descriptions_reject_non_string_entries(prop_desc):
    def helper(value: str) -> str:
        return value

    with pytest.raises(TypeError, match="prop_desc keys and values"):
        Function.from_callable(helper, prop_desc=prop_desc)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="prop_desc keys and values"):
        tool(prop_desc=prop_desc)(helper)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_native_function_rejects_coerced_boolean_flags(value):
    with pytest.raises(ValidationError):
        Function(
            name="collect",
            description="Collect.",
            properties={},
            strict=value,  # type: ignore[arg-type]
        )

    with pytest.raises(ValidationError):
        Function(
            name="collect",
            description="Collect.",
            properties={},
            additional_properties=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_native_function_helpers_reject_coerced_strict_flag(value):
    def helper(value: str) -> str:
        return value

    with pytest.raises(TypeError, match="strict"):
        Function.from_callable(helper, strict=value)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="strict"):
        tool(strict=value)  # type: ignore[arg-type]


def test_native_function_schema_views_are_isolated():
    properties = {"left": {"type": "integer"}}
    required = ["left"]
    function = Function(
        name="add",
        description="Add one value",
        properties=properties,
        required=required,
        function=lambda left: left + 1,
    )
    prompt = Prompt(path="tools", prompt="Use tools.", function_list=[function])

    properties["left"]["type"] = "string"
    required.append("right")
    tool_schema = function.to_tool()
    tool_schema["function"]["parameters"]["properties"]["left"]["type"] = "number"
    tool_schema["function"]["parameters"]["required"].append("right")
    prompt_functions = prompt.functions
    prompt_functions["add"].properties["left"]["type"] = "boolean"
    prompt_function = prompt.get_function("add")

    assert function.properties == {"left": {"type": "integer"}}
    assert function.required == ["left"]
    assert prompt_function.properties == {"left": {"type": "integer"}}
    assert prompt_function(FunctionCall(name="add", arguments={"left": 2})).result == 3


def test_native_message_models_isolate_mutable_constructor_inputs():
    arguments = {"items": [1]}
    result = {"values": [1]}
    call = FunctionCall(name="collect", arguments=arguments, result=result)
    content = [{"text": ["hello"]}]
    logprob_bytes = [1, 2]
    child_logprob = TokenLogprob(token="h", bytes=[3])
    logprob = TokenLogprob(
        token="hello",
        bytes=logprob_bytes,
        top_logprobs=[child_logprob],
    )
    parsed = {"labels": ["greeting"]}
    usage = {"nested": {"tokens": [1]}}
    metadata = {"trace": {"id": "one"}}
    vectors = [1.0]
    message = Message(
        role=Role.USER,
        content=content,
        name="user",
        function_calls=[call],
        logprobs=[logprob],
        parsed=parsed,
        usage=usage,
        metadata=metadata,
        vectors=vectors,
    )

    arguments["items"].append(2)
    result["values"].append(2)
    content[0]["text"].append("changed")
    logprob_bytes.append(4)
    child_logprob.bytes.append(5)
    parsed["labels"].append("changed")
    usage["nested"]["tokens"].append(2)
    metadata["trace"]["id"] = "two"
    vectors.append(2.0)

    assert call.arguments == {"items": [1]}
    assert call.result == {"values": [1]}

    call.arguments["items"].append(3)
    logprob.top_logprobs[0].bytes.append(6)

    assert message.content == [{"text": ["hello"]}]
    assert message.function_calls[0].arguments == {"items": [1]}
    assert message.logprobs[0].bytes == [1, 2]
    assert message.logprobs[0].top_logprobs[0].bytes == [3]
    assert message.parsed == {"labels": ["greeting"]}
    assert message.usage == {"nested": {"tokens": [1]}}
    assert message.metadata == {"trace": {"id": "one"}}
    assert message.vectors == [1.0]


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        ("FunctionCall.id", lambda: FunctionCall(id=b"call-1", name="collect")),
        ("FunctionCall.name", lambda: FunctionCall(name=b"collect")),
        (
            "FunctionCall.result_str",
            lambda: FunctionCall(name="collect", result_str=b"ok"),
        ),
        (
            "FunctionCall.error_message",
            lambda: FunctionCall(name="collect", error_message=b"failed"),
        ),
        ("TokenLogprob.token", lambda: TokenLogprob(token=b"hello")),
        (
            "Message.content",
            lambda: Message(role=Role.USER, content=b"hello", name="user"),
        ),
        (
            "Message.name",
            lambda: Message(role=Role.USER, content="hello", name=b"user"),
        ),
        (
            "Message.model",
            lambda: Message(
                role=Role.USER,
                content="hello",
                name="user",
                model=b"gpt",
            ),
        ),
        (
            "Function.name",
            lambda: Function(name=b"collect", description="Collect.", properties={}),
        ),
        (
            "Function.description",
            lambda: Function(name="collect", description=b"Collect.", properties={}),
        ),
        (
            "Function.required",
            lambda: Function(
                name="collect",
                description="Collect.",
                properties={},
                required=[b"value"],
            ),
        ),
        ("Prompt.path", lambda: Prompt(path=b"prompt", prompt="Hello.")),
        ("Prompt.prompt", lambda: Prompt(path="prompt", prompt=b"Hello.")),
        (
            "Prompt.function_list",
            lambda: Prompt(path="prompt", prompt="Hello.", function_list=[b"collect"]),
        ),
    ],
)
def test_native_models_reject_bytes_for_text_fields(label, factory):
    with pytest.raises(ValidationError) as exc_info:
        factory()

    assert any(error["type"] == "string_type" for error in exc_info.value.errors()), label


def test_dialog_put_prompt_fork_and_roundtrip_lineage():
    prompt = Prompt(path="system", prompt="You are {name}.")
    dialog = Dialog(owner="agent")

    first = dialog.put_prompt(prompt, prompt_args={"name": "careful"}, role=Role.SYSTEM)
    dialog.put_text("hello", name="operator")
    child = dialog.fork(last_n=1, first_k=1)

    assert first.metadata["dialog_id"] == dialog.dialog_id
    assert child.parent is dialog
    assert child.depth == 1
    assert dialog.children == [child]
    assert len(child.messages) == 2
    assert child.tree_node.parent_id == dialog.dialog_id
    assert child.tree_node.last_n == 1
    assert dialog.tree_node.children_ids == [child.dialog_id]

    restored = Dialog.from_dict(dialog.to_dict())

    assert restored.owner == "agent"
    assert restored.head.content == "You are careful."
    assert restored.tail.content == "hello"
    assert len(restored.children) == 1
    restored_child = restored.children[0]
    assert restored_child.parent is restored
    assert restored_child.depth == 1
    assert restored_child.tree_node.parent_id == restored.dialog_id
    assert restored_child.messages[-1].content == "hello"


def test_native_prompt_and_message_metadata_are_isolated():
    prompt_metadata = {"nested": {"value": 1}}
    addon_args = {"settings": {"tone": "careful"}}
    prompt = Prompt(
        path="draft",
        prompt="Write about {topic}.",
        metadata=prompt_metadata,
        addon_args=addon_args,
    )
    info = prompt.info_dict()
    child = prompt.extend(path="draft/child")
    dialog = Dialog()
    message_metadata = {"nested": {"value": 1}}
    message = dialog.put_text("hello", metadata=message_metadata)
    prompt_message_metadata = {"nested": {"value": 2}}
    prompt_message = dialog.put_prompt(
        prompt,
        prompt_args={"topic": "metadata"},
        metadata=prompt_message_metadata,
    )

    prompt_metadata["nested"]["value"] = 99
    addon_args["settings"]["tone"] = "changed"
    info["metadata"]["nested"]["value"] = 100
    info["addon_args"]["settings"]["tone"] = "loud"
    prompt.metadata["nested"]["value"] = 3
    prompt.addon_args["settings"]["tone"] = "brief"
    message_metadata["nested"]["value"] = 8
    prompt_message_metadata["nested"]["value"] = 9

    assert prompt.metadata == {"nested": {"value": 3}}
    assert prompt.addon_args == {"settings": {"tone": "brief"}}
    assert child.metadata == {"nested": {"value": 1}}
    assert child.addon_args == {"settings": {"tone": "careful"}}
    assert message.metadata["nested"] == {"value": 1}
    assert prompt_message.metadata["nested"] == {"value": 2}


def test_native_models_accept_nested_read_only_mapping_metadata():
    prompt_inner = {"value": 1}
    addon_inner = {"tone": "careful"}
    message_inner = {"value": 2}
    prompt_message_inner = {"value": 3}
    prompt = Prompt(
        path="draft",
        prompt="Write about {topic}.",
        metadata={"nested": MappingProxyType(prompt_inner)},
        addon_args={"settings": MappingProxyType(addon_inner)},
    )
    dialog = Dialog()
    message = dialog.put_text(
        "hello",
        metadata={"nested": MappingProxyType(message_inner)},
    )
    prompt_message = dialog.put_prompt(
        prompt,
        prompt_args={"topic": "metadata"},
        metadata={"nested": MappingProxyType(prompt_message_inner)},
    )

    prompt_inner["value"] = 10
    addon_inner["tone"] = "changed"
    message_inner["value"] = 20
    prompt_message_inner["value"] = 30

    assert prompt.info_dict()["metadata"] == {"nested": {"value": 1}}
    assert prompt.info_dict()["addon_args"] == {"settings": {"tone": "careful"}}
    assert message.metadata["nested"] == {"value": 2}
    assert prompt_message.metadata["nested"] == {"value": 3}


@pytest.mark.parametrize("metadata", [[], [("owner", "tests")], "bad", 123])
def test_native_dialog_rejects_non_mapping_message_metadata(metadata):
    dialog = Dialog()
    prompt = Prompt(path="draft", prompt="Write.")

    with pytest.raises(TypeError, match="metadata"):
        dialog.put_text("hello", metadata=metadata)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="metadata"):
        dialog.put_prompt(prompt, metadata=metadata)  # type: ignore[arg-type]


@pytest.mark.parametrize("prompt_args", [[], [("topic", "tests")], "bad", 123])
def test_dialog_put_prompt_rejects_non_mapping_prompt_args(prompt_args):
    dialog = Dialog()
    prompt = Prompt(path="draft", prompt="Write about {topic}.")

    with pytest.raises(TypeError, match="prompt_args"):
        dialog.put_prompt(prompt, prompt_args=prompt_args)  # type: ignore[arg-type]


def test_default_tag_parser_extracts_prompt_outputs():
    parser = DefaultTagParser(
        xml_tags=["answer"],
        md_tags=["json"],
        signal_tags=["DONE"],
        required_xml_tags=["answer"],
        required_md_tags=["json"],
    )
    prompt = Prompt(path="parse", prompt="Parse output.", parser=parser)
    content = """
<answer>Hello</answer>
```json
{"ok": true}
```
<DONE>
""".strip()

    parsed = prompt.parse(content)

    assert parsed["xml_tags"]["answer"] == ["Hello"]
    assert parsed["md_tags"]["json"] == ['{"ok": true}']
    assert parsed["signal_tags"]["DONE"] is True
    assert find_xml_blocks(content, "answer") == ["Hello"]
    assert find_md_blocks(content, "json") == ['{"ok": true}']


def test_prompt_parse_does_not_probe_parser_parse_property():
    class CallableParser:
        @property
        def parse(self):
            raise AssertionError("parse property should not be evaluated")

        def __call__(self, content, **runtime_args):
            return {"content": content, "runtime_args": runtime_args}

    prompt = Prompt(path="parse", prompt="Parse output.", parser=CallableParser())

    assert prompt.parse("hello", mode="strict") == {
        "content": "hello",
        "runtime_args": {"mode": "strict"},
        "raw": "hello",
    }


def test_prompt_parse_accepts_classmethod_parse_parser():
    class ClassMethodParser:
        @classmethod
        def parse(cls, content, **runtime_args):
            return {"parser": cls.__name__, "content": content, **runtime_args}

    prompt = Prompt(path="parse", prompt="Parse output.", parser=ClassMethodParser())

    assert prompt.parse("hello", mode="strict") == {
        "parser": "ClassMethodParser",
        "content": "hello",
        "mode": "strict",
        "raw": "hello",
    }


def test_default_tag_parser_reports_missing_required_blocks():
    parser = DefaultTagParser(required_xml_tags=["answer"], required_md_tags=["json"])

    with pytest.raises(ParseError) as exc_info:
        parser.parse("<answer>Hello</answer>")

    assert "Missing required markdown block" in str(exc_info.value)
