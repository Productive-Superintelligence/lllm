# Parallel Search MCP tactic

This example attaches `web_search` and `web_fetch` from the anonymous
[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp) to
Pydantic AI, then exposes the agent through LLLM's `PydanticAITactic`.
Pydantic AI owns tool discovery, dispatch and model execution. The example uses
Streamable HTTP at `https://search.parallel.ai/mcp` and sends a project
User-Agent. No Parallel API key is read or sent.

From the repository root, install in a fresh environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e . -r examples/parallel_search/requirements.txt
```

Try real search and fetch without model credentials:

```bash
.venv/bin/python examples/parallel_search/agent.py --smoke
```

The smoke mode uses Pydantic AI's deterministic `FunctionModel` to dispatch
both tools through `PydanticAITactic.arun()` and prints their results. It makes
network requests, but does not call an inference provider or generate an answer.

For a model-generated answer, select a model and provide its credentials:

```bash
export OPENAI_API_KEY=your-model-provider-key
.venv/bin/python examples/parallel_search/agent.py --model openai:gpt-4.1-mini \
  "What is LLLM's Tactic protocol? Include source URLs."
```

The declared dependencies include the OpenAI provider. Other Pydantic AI
providers need their own optional dependencies and credentials. Model inference
is separate from Parallel's free, rate-limited search tier. Anonymous searches
use fast mode; excerpts are capped to roughly 25,000 characters per tool call.
Network errors and rate limits propagate to the caller.

To use this in an application, import `build_tactic(model)` from `agent.py`,
enter `async with tactic.agent`, then call `await tactic.arun(question)`.
Keep the agent context open for related calls and close it when finished.
This example changes no LLLM defaults or existing provider configuration.
