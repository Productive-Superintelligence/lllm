import pytest

from lllm import DefaultTagParser, ParseError, find_md_blocks, find_xml_blocks


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
