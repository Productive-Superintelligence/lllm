# Agents

An agent is a model plus the loop that calls it. It keeps its own dialogs,
runs the tools the model asks for, repairs replies that fail to parse, and
records what every call cost. Your program decides *what* to ask; the agent
makes each ask reliable.

| Part | What it is |
| --- | --- |
| `model`, `model_args` | Any [LiteLLM](models.md) model name, and settings such as `temperature` |
| `system_prompt` | A [prompt](prompts.md) that starts every dialog the agent opens |
| Dialogs | The agent's conversations, each under an alias ([Dialogs](dialogs.md)) |
| Tools | Functions the model can call, from prompts or the agent's config ([Tools](tools.md)) |
| Limits | How many tool rounds, repairs, and provider retries one call may use |

Here is an agent holding a conversation. It remembers the first turn because
both turns go into the same dialog:

```python title="chat.py"
from lllm import Tactic

agent = Tactic.quick(system_prompt="You are a helpful assistant.", model="gpt-4o-mini")
agent.open("chat")

agent.receive("My name is Ada.")
print(agent.respond().content)
#> Nice to meet you, Ada!

agent.receive("What is my name?")
print(agent.respond().content)
#> Your name is Ada.
```

## Creating agents

There are three ways to get an agent, from quickest to most structured.

**`Tactic.quick()`** builds a one-off agent. With a question, it returns the
reply; without one, it returns the agent:

```python title="quick.py"
from lllm import Tactic

reply = Tactic.quick("What is the capital of France?", model="gpt-4o-mini")
print(reply.content)
#> The capital of France is Paris.

agent = Tactic.quick(system_prompt="Answer in one word.", model="gpt-4o-mini")
print(agent.name, agent.model)
#> assistant gpt-4o-mini
```

**In a tactic**, agents are declared by name and described by config. Every
call of the tactic gets fresh agents, so calls never share state. This is the
usual way to use agents; see [Tactics](tactics.md):

```python test="skip" reason="fragment"
class Review(Tactic):
    agent_group = ["reviewer"]

    def call(self, code: str) -> str:
        reviewer = self.agents["reviewer"]
        ...

review = Review({
    "global": {"model_name": "gpt-4o-mini"},
    "agent_configs": [{"name": "reviewer", "system_prompt": "You review code."}],
})
```

**Directly**, `Agent` is a dataclass you can build yourself, for example in a
script or a notebook:

```python title="direct.py"
from lllm import Agent, Prompt
from lllm.runtimes.native.invokers import build_invoker

agent = Agent(
    name="writer",
    system_prompt=Prompt(path="writer/system", prompt="You write clear notes."),
    model="gpt-4o-mini",
    llm_invoker=build_invoker({"invoker": "litellm"}),
    model_args={"temperature": 0.2},
)
agent.open("draft")
agent.receive("Write a one-line project update.")
print(agent.respond().content)
#> The docs now run every example and check its output.
```

## Conversations

An agent can keep several dialogs at once, each under an alias, and works on
the active one. Use one alias per line of work, so histories never mix:

```python title="aliases.py"
from lllm import Tactic

agent = Tactic.quick(system_prompt="You are a careful planner.", model="gpt-4o-mini")

agent.open("plan")  # (1)!
agent.receive("Plan a launch checklist in three steps.")
plan = agent.respond().content

agent.open("risks")  # (2)!
agent.receive(f"List the biggest risk in this plan:\n{plan}")
print(agent.respond().content)
#> The biggest risk is skipping the rollback test.

agent.switch("plan")  # (3)!
print(agent.active_alias, list(agent.dialogs))
#> plan ['plan', 'risks']
```

1. `open()` starts a dialog with the agent's system prompt and makes it active.
2. Opening a second dialog switches to it; the first keeps its history.
3. `switch()` goes back. `close(alias)` removes a dialog, and
   `fork(alias, child_alias)` branches one ([Dialogs](dialogs.md#forking)).

| Method | Does |
| --- | --- |
| `open(alias, prompt_args=None)` | Start a dialog; `prompt_args` fill the system prompt's variables |
| `receive(text)` | Add a user message |
| `receive_prompt(prompt, args)` | Render a [prompt](prompts.md) and add it, with its format, parser, and tools |
| `receive_image(image, caption=None)` | Add an image (a path, URL, bytes, or PIL image) |
| `respond(args=None, return_session=False)` | Run the agent loop on the active dialog |
| `await arespond(...)` | The same, for async code |

## Responses

`respond()` returns the final `Message`. Its `content` is the text, and
`parsed` holds structured data when the prompt asked for a
[format](prompts.md#structured-output). Ask for the call session instead to
see how the answer was reached:

```python title="session.py"
from pydantic import BaseModel

from lllm import Prompt, Tactic


class City(BaseModel):
    name: str
    country: str


ASK = Prompt(path="geo/capital", prompt="Name the capital of {country}.", format=City)

agent = Tactic.quick(system_prompt="You answer geography questions.", model="gpt-4o-mini")
agent.open("q")
agent.receive_prompt(ASK, {"country": "Japan"})
session = agent.respond(return_session=True)

print(session.state, session.delivery.parsed)
#> success {'name': 'Tokyo', 'country': 'Japan'}
print(session.exception_retries_count)  # (1)!
#> 1
```

1. The model's first reply did not match `City`; the agent sent the error
   back, and the second reply parsed. One repair, no code on your side.

An `AgentCallSession` records every model call (`invoke_traces`), each
repair (`exception_retries`), each tool round (`interrupts`), provider
retries (`llm_recalls`), and the final `delivery`. Inside a tactic, every
agent call is also added to the tactic's session, with its cost.

## The agent loop

`respond()` repeats until the model gives a final answer:

1. Call the model with the dialog, the prompt's tools, and `model_args`.
2. If the reply calls tools, run them, add the results, and go to 1.
3. If the reply fails to parse (the prompt's parser or `format=`), add the
   prompt's repair message (`Error: ... Please fix.`) and go to 1.
4. Otherwise add the reply to the dialog and return it.

Repairs and tool calls happen in a working copy, so the dialog you keep holds
the clean exchange. Every branch has a budget:

| Setting | Default | Limits |
| --- | --- | --- |
| `max_interrupt_steps` | 5 | Tool rounds per call. After the last one the model is asked to answer without tools. `0` means up to 100. |
| `max_exception_retry` | 3 | Repairs of unparsable replies, then `AgentException` is raised. |
| `max_llm_recall` | 0 | Retries after a provider error. |
| `max_rate_limit_retries` | 8 | Retries after rate limits, waiting `min(60, 2**n)` seconds plus jitter. |

Set them in the agent's config (or per agent in a tactic's
`agent_configs`). To change what the repair or tool-result messages say,
give the prompt a [handler](prompts.md#handlers).

## Model settings

`model_args` are sent with every call: `temperature`, `max_tokens`, `seed`,
a provider's `api_base`, and anything else LiteLLM accepts. Override them for
one call with `respond(args=...)`:

```python title="settings.py"
from lllm import Tactic

agent = Tactic.quick(
    system_prompt="You name things.", model="gpt-4o-mini", temperature=0.2
)
agent.open("names")
agent.receive("Suggest a name for a caching library.")
print(agent.respond(args={"temperature": 1.0}).content)
#> Stash
```

[Models](models.md) covers providers, local servers, and custom invokers, and
[Inference hints](inference-hints.md) covers session-aware serving.

## Async

`arespond()` runs the same loop without blocking the event loop, and inside a
tactic it is recorded like `respond()`:

```python title="async_agent.py"
import asyncio

from lllm import Tactic


async def main():
    agent = Tactic.quick(system_prompt="You are terse.", model="gpt-4o-mini")
    agent.open("chat")
    agent.receive("What is 2 + 2?")
    reply = await agent.arespond()
    print(reply.content)
    #> 4


asyncio.run(main())
```

## Tools and skills

An agent's tools come from three places: the prompt driving the turn
(`function_list`), the agent's system prompt, and its config (`tools:`,
proxies, and skills). All of them stay available on every turn unless a
prompt sets `agent_tools=False`. See [Tools](tools.md).
