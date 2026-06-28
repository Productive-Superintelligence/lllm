import pytest

from lllm import (
    DefaultTagParser,
    ParseError,
    Tactic,
    find_md_blocks,
    find_xml_blocks,
)


def test_default_tag_parser_is_runtime_agnostic():
    parser = DefaultTagParser(
        required_xml_tags=["answer"],
        required_md_tags=["json"],
        signal_tags=["DONE"],
    )
    content = "<answer>Hello</answer>\n```json\n{}\n```\n<DONE>"

    parsed = parser.parse(content)

    assert parsed["xml_tags"]["answer"] == ["Hello"]
    assert parsed["md_tags"]["json"] == ["{}"]
    assert parsed["signal_tags"]["DONE"] is True
    assert find_xml_blocks(content, "answer") == ["Hello"]
    assert find_md_blocks(content, "json") == ["{}"]


def test_default_tag_parser_raises_shared_parse_error():
    parser = DefaultTagParser(required_xml_tags=["answer"])

    with pytest.raises(ParseError):
        parser.parse("missing")


def test_default_tag_parser_exports_schema_and_round_trips_config():
    parser = DefaultTagParser(
        required_xml_tags=["answer"],
        required_md_tags=["json"],
        signal_tags=["DONE"],
        parser_args={"mode": "strict"},
    )

    schema = DefaultTagParser.model_json_schema()
    restored = DefaultTagParser.model_validate(parser.model_dump(mode="json"))

    assert schema["title"] == "DefaultTagParser"
    assert "required_xml_tags" in schema["properties"]
    assert "required_md_tags" in schema["properties"]
    assert "signal_tags" in schema["properties"]
    assert restored.required_xml_tags == ["answer"]
    assert restored.required_md_tags == ["json"]
    assert restored.signal_tags == ["DONE"]
    assert restored.parser_args == {"mode": "strict"}


def test_default_tag_parser_can_back_plain_tactic_boundary():
    parser = DefaultTagParser(required_xml_tags=["answer"], signal_tags=["DONE"])

    class ParseAnswerTactic(Tactic[str, dict[str, object]]):
        name = "parse_answer"
        input_type = str
        output_type = dict[str, object]

        def _run(self, input_value, *, context=None):
            return parser.parse(input_value)

    tactic = ParseAnswerTactic()

    parsed = tactic.run("<answer>Hello</answer><DONE>")

    assert parsed["xml_tags"]["answer"] == ["Hello"]
    assert parsed["signal_tags"]["DONE"] is True
    assert tactic.info().input_schema == {"type": "string"}
    assert tactic.info().output_schema is not None
