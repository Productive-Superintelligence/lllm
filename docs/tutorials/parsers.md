# Parsers

Goal: parse structured model or tactic output without depending on a runtime.

```python
from lllm.parsers import DefaultTagParser


parser = DefaultTagParser(
    required_xml_tags=["answer"],
    required_md_tags=["json"],
    signal_tags=["DONE"],
)

parsed = parser.parse("<answer>Hello</answer>\n```json\n{}\n```\n<DONE>")
assert parsed["xml_tags"]["answer"] == ["Hello"]
assert parsed["signal_tags"]["DONE"] is True
```

## Contract

Parser configuration is Pydantic-schema-compatible, so it can be serialized in
runtime config or inspected by tools:

```python
schema = DefaultTagParser.model_json_schema()
assert "required_xml_tags" in schema["properties"]
```

The parser fails fast with `ParseError` when required blocks are missing.
Repair loops, retries, or partial-output recovery should live in the tactic or
runtime adapter that owns the model call.

## In A Tactic

The same parser can be attached to native prompts or used directly around
Pydantic AI/plain-Python tactic outputs:

```python
from lllm import Tactic
from lllm.parsers import DefaultTagParser


parser = DefaultTagParser(required_xml_tags=["answer"], signal_tags=["DONE"])


class ParseAnswerTactic(Tactic[str, dict[str, object]]):
    name = "parse_answer"
    input_type = str
    output_type = dict[str, object]

    def _run(self, input_value, *, context=None):
        return parser.parse(input_value)


parsed = ParseAnswerTactic().run("<answer>Hello</answer><DONE>")
assert parsed["xml_tags"]["answer"] == ["Hello"]
```
