"""Vendor tool prompts — registered under the 'tools' prefix."""
from lllm.core.prompt import Prompt

web_search = Prompt(
    path="web_search",
    prompt="Search the web for: {query}\n\nReturn the top {n} results.",
    metadata={"tool": "web_search"},
)

calculator = Prompt(
    path="calculator",
    prompt="Calculate the result of: {expression}\n\nReturn just the numeric answer.",
    metadata={"tool": "calculator"},
)

translator = Prompt(
    path="translator",
    prompt="Translate the following text from {source_lang} to {target_lang}:\n\n{text}",
    metadata={"tool": "translator"},
)
