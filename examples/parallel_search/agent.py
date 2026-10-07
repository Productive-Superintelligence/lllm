"""Parallel Search MCP tools in a Pydantic AI agent, exposed as an LLLM tactic."""

import argparse
import asyncio
import json
import uuid

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from lllm.runtimes import PydanticAITactic

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = (
    "lllm-core/parallel-search-example "
    "(https://github.com/Productive-Superintelligence/lllm)"
)


def build_tactic(model) -> PydanticAITactic:
    """Keep model selection and MCP execution in Pydantic AI."""
    server = MCPToolset(
        MCP_URL,
        headers={"User-Agent": USER_AGENT},
        init_timeout=30,
        read_timeout=60,
    )
    agent = Agent(
        model,
        name="parallel-search",
        toolsets=[server],
        instructions="Use web_search and web_fetch to research the question. Cite source URLs.",
    )
    return PydanticAITactic(agent, input_type=str, output_type=str)


def smoke_model() -> FunctionModel:
    """Dispatch real MCP tools without an inference provider or model credentials."""
    session_id = uuid.uuid4().hex

    def respond(messages, info):
        for message in messages:
            for part in message.parts:
                if isinstance(part, RetryPromptPart):
                    raise RuntimeError(f"Smoke tool call failed: {part.content}")
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("web_search", {
                "objective": "Find the official LLLM Python project and its tactic protocol.",
                "search_queries": ["Productive Superintelligence LLLM tactics"],
                "session_id": session_id,
            })])
        if len(returns) == 1:
            return ModelResponse(parts=[ToolCallPart("web_fetch", {
                "urls": ["https://github.com/Productive-Superintelligence/lllm"],
                "objective": "Describe the Tactic protocol and Pydantic AI integration.",
                "session_id": session_id,
            })])
        return ModelResponse(parts=[TextPart(json.dumps(
            {part.tool_name: part.content for part in returns}, indent=2
        ))])

    return FunctionModel(respond)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model", help="Pydantic AI model name, e.g. openai:gpt-4.1-mini")
    choice.add_argument("--smoke", action="store_true", help="Real search/fetch with no model API")
    parser.add_argument("question", nargs="?", default="What is LLLM's Tactic protocol?")
    args = parser.parse_args()
    tactic = build_tactic(smoke_model() if args.smoke else args.model)
    async with tactic.agent:
        print(await tactic.arun(args.question))


if __name__ == "__main__":
    asyncio.run(main())
