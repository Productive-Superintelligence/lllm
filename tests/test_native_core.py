import pytest

from lllm.runtimes.native import (
    DefaultTagParser,
    Dialog,
    FunctionCall,
    ParseError,
    Prompt,
    Role,
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


def test_default_tag_parser_reports_missing_required_blocks():
    parser = DefaultTagParser(required_xml_tags=["answer"], required_md_tags=["json"])

    with pytest.raises(ParseError) as exc_info:
        parser.parse("<answer>Hello</answer>")

    assert "Missing required markdown block" in str(exc_info.value)
