"""Analysis prompts with structured output parsers."""
from lllm.core.prompt import Prompt, DefaultTagParser

summary = Prompt(
    path="summary",
    prompt=(
        "Analyze the following text and provide a structured summary:\n\n"
        "{text}\n\n"
        "Provide your response in <summary>...</summary> tags with key points in <key_points>...</key_points>."
    ),
    parser=DefaultTagParser(
        xml_tags=["summary", "key_points"],
        required_xml_tags=["summary"],
    ),
    metadata={"type": "analysis", "version": "2.0"},
)

multi_tag = Prompt(
    path="multi_tag",
    prompt=(
        "Analyze {subject}.\n"
        "Provide:\n"
        "<strengths>...</strengths>\n"
        "<weaknesses>...</weaknesses>\n"
        "<recommendations>...</recommendations>"
    ),
    parser=DefaultTagParser(
        xml_tags=["strengths", "weaknesses", "recommendations"],
        required_xml_tags=["strengths", "weaknesses"],
    ),
    metadata={"type": "swot"},
)

code_review = Prompt(
    path="code_review",
    prompt=(
        "Review the following {language} code:\n\n{code}\n\n"
        "Provide a code review with issues and suggestions."
    ),
    parser=DefaultTagParser(
        xml_tags=["issues", "suggestions"],
        signal_tags=["LGTM", "NEEDS_CHANGES"],
    ),
    metadata={"type": "code_review"},
)

signal_only = Prompt(
    path="signal_only",
    prompt="Evaluate: {claim}\n\nRespond with <VALID> if valid, <INVALID> if not.",
    parser=DefaultTagParser(
        signal_tags=["VALID", "INVALID"],
    ),
    metadata={"type": "validation"},
)
