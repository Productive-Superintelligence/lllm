"""Agent prompts for pkg_delta."""
from lllm.core.prompt import Prompt

search_agent = Prompt(
    path="search_agent",
    prompt="Search for information about {query} and summarize the top results.",
    metadata={"uses_tools": True},
)

index_agent = Prompt(
    path="index_agent",
    prompt="Index the following document: {document}",
)
