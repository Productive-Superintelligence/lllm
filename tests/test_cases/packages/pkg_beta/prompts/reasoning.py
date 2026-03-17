"""Reasoning prompts for pkg_beta — chain-of-thought and debate prompts."""
from lllm.core.prompt import Prompt, DefaultTagParser

chain_of_thought = Prompt(
    path="chain_of_thought",
    prompt=(
        "Think step-by-step about:\n\n{problem}\n\n"
        "Show your reasoning as <thinking>...</thinking>, "
        "then give your final answer as <answer>...</answer>."
    ),
    parser=DefaultTagParser(
        xml_tags=["thinking", "answer"],
        required_xml_tags=["answer"],
    ),
    metadata={"type": "cot"},
)

socratic = Prompt(
    path="socratic",
    prompt="Ask a series of 5 Socratic questions about: {topic}",
    metadata={"type": "socratic"},
)

debate = Prompt(
    path="debate",
    prompt=(
        "Present balanced arguments for and against:\n\n{proposition}\n\n"
        "For: <pro>...</pro>\nAgainst: <con>...</con>"
    ),
    parser=DefaultTagParser(xml_tags=["pro", "con"]),
    metadata={"type": "debate"},
)

hypothetical = Prompt(
    path="hypothetical",
    prompt="If {condition} were true, what would be the implications for {domain}?",
    metadata={"type": "hypothetical"},
)
