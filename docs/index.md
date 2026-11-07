# Low-Level Language Models

<p class="psi-brand">
  <img src="assets/lllm-logo-text-dark.png#only-light" alt="LLLM" />
  <img src="assets/lllm-logo-text-white.png#only-dark" alt="LLLM" />
</p>

[lllm.one](https://lllm.one){ .psi-domain }

**LLLM** is a Python framework for agentic systems that do real work: they
break hard questions down, call tools, write and run code, and need to be
inspected turn by turn when something goes wrong. Instead of hiding the agent
loop behind a single `run()`, LLLM gives you one small, plain object for each
part of it and lets you write the program that connects them.

## Why LLLM

- **You write the program.** A [tactic](tactics.md) is ordinary Python that
  calls agents. Loops, branches, fan-out, and retries are your code, not a
  graph language.
- **Every turn is a function you can read.** A [prompt](prompts.md) carries
  its template, output format, parser, tools, and error handlers. The agent
  loop retries malformed answers and runs tool calls for you.
- **Agents have memory you can inspect.** Each agent keeps its
  [dialogs](dialogs.md) as trees you can fork, replay, save, and load.
- **Typed end to end.** Inputs, outputs, and tool arguments are
  [Pydantic](typed-io.md) models, and every tactic publishes JSON schemas.
- **Ready for production.** One command [serves](serving.md) a tactic as a
  FastAPI app, [packages](packages.md) are shared on the
  [hub](https://lllm.one/hub/), and agents send
  [inference hints](inference-hints.md) to KV-cache-aware servers such as
  NVIDIA Dynamo, vLLM, and SGLang.
- **Works with what you have.** Any model [LiteLLM](models.md) supports,
  local model servers, plain Python functions, and
  [Pydantic AI](pydantic-ai.md) agents.

## Hello world

```python title="hello_world.py"
from lllm import Tactic

print(Tactic.quick("Where does 'hello world' come from?").content)
#> It comes from Brian Kernighan's 1972 tutorial for the B language.
```

`Tactic.quick()` builds a one-off agent, sends it the question, and returns
the reply. It uses `gpt-4o` by default; [install](install.md) LLLM, set
`OPENAI_API_KEY`, or pass `model=` to use any other provider.

## A real program: two agents, a tool, and typed output

Here is the shape of most LLLM programs. A tactic owns two agents: a
researcher that searches a knowledge base with a tool, and a writer that
turns the notes into a typed brief.

```python title="brief.py"
from pydantic import BaseModel

from lllm import Prompt, Tactic, tool

KNOWLEDGE = {
    "kv caches": "Transformers cache attention keys and values so each new "
    "token reuses earlier work instead of recomputing it.",
}


@tool(description="Search the knowledge base for a topic")
def search(query: str) -> str:  # (1)!
    return KNOWLEDGE.get(query.lower(), "No results.")


class Brief(BaseModel):  # (2)!
    title: str
    bullets: list[str]


RESEARCH = Prompt(  # (3)!
    path="brief/research",
    prompt="Collect facts about {topic}.",
    function_list=[search],
)
WRITE = Prompt(
    path="brief/write",
    prompt="Write a three-bullet brief about {topic} from these notes:\n{notes}",
    format=Brief,
)


class BriefTactic(Tactic):  # (4)!
    name = "brief"
    agent_group = ["researcher", "writer"]
    output_model = Brief

    def call(self, topic: str) -> Brief:  # (5)!
        researcher = self.agents["researcher"]
        researcher.open("research")
        researcher.receive_prompt(RESEARCH, {"topic": topic})
        notes = researcher.respond().content  # (6)!

        writer = self.agents["writer"]
        writer.open("draft")
        writer.receive_prompt(WRITE, {"topic": topic, "notes": notes})
        return writer.respond().parsed  # (7)!


brief = BriefTactic(
    {
        "global": {"model_name": "gpt-4o-mini"},  # (8)!
        "agent_configs": [
            {"name": "researcher", "system_prompt": "You research topics."},
            {"name": "writer", "system_prompt": "You write crisp briefs."},
        ],
    }
)

result = brief("KV caches")
print(result.title)
#> KV caches in one minute
print("\n".join(f"- {bullet}" for bullet in result.bullets))
"""
- A KV cache stores attention keys and values from earlier tokens.
- Reusing it makes each new token cost one step, not a full pass.
- Shared prompt prefixes can reuse cached entries across requests.
"""
```

1. A [tool](tools.md) is a typed Python function. LLLM builds its JSON schema
   from the annotations, and the agent loop runs it when the model asks.
2. Outputs are Pydantic models, so the rest of your program gets real objects.
3. A [prompt](prompts.md) is a function: a template plus everything one turn
   needs. `RESEARCH` brings its own tool; `WRITE` demands a `Brief`, and the
   agent re-asks the model if the reply does not parse.
4. A [tactic](tactics.md) declares the agents it uses. Every call gets fresh
   agents built from the config, so calls never share state.
5. `call()` is the program: plain Python, with control flow you can step
   through in a debugger.
6. `respond()` runs the agent loop: the model calls `search`, LLLM runs it,
   sends the result back, and returns the final message.
7. `parsed` holds the structured reply. Because the tactic declares
   `output_model = Brief`, its result is validated into a `Brief`.
8. Models are configured, not hard-coded. Any LiteLLM model name works, per
   tactic or per agent.

The result is an object you can use, and the whole run stays inspectable:
call `brief(topic, return_session=True)` to get every agent call, its cost,
and each agent's dialog.

## Serve it

Every tactic is also a typed web service. Point `lllm serve` at it:

```bash
lllm serve brief.py:brief --port 8000
curl -X POST localhost:8000/run -H 'content-type: application/json' \
  -d '{"input": "KV caches"}'
```

You get `POST /run`, `POST /stream`, `GET /info`, and OpenAPI docs with the
input and output schemas. See [Serving](serving.md).

## How the pieces fit

LLLM treats an agentic system as a program, built from four objects:

- **Tactic is the program.** It owns a typed interface and the control flow.
- **Agent is the caller.** It runs the loop that calls the model, repairs
  malformed answers, and runs tools.
- **Prompt is the function.** It specifies one turn: template, output format,
  parser, tools, and handlers.
- **Dialog is the mental state.** It holds what one agent has seen and said,
  as a tree.

[How LLLM works](concepts.md) explains the model in one page, and each
concept has its own guide.

## Next steps

- [Install LLLM](install.md) and connect a model.
- Read the core concepts: [agents](agents.md), [prompts](prompts.md),
  [tools](tools.md), [dialogs](dialogs.md), and [tactics](tactics.md).
- [Test your agents](testing.md) without calling a model.
- Browse complete [examples](examples/index.md).

LLLM's design grew out of the reasoning units of Analytica (ICLR 2026), which
breaks questions into grounded subpropositions and runs specialized agents
over them, and it has since served autonomous software development in
Apeiron (Findings of ACL 2026).
