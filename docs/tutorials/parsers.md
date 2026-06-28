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

The same parser can be attached to native prompts or used directly around
Pydantic AI/plain-Python tactic outputs.
