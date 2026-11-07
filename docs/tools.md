# Tools

A tool is a typed Python function the model can call. LLLM builds its JSON
schema from the type annotations, offers it to the model, and when the model
asks for it, the agent loop runs your function and sends the result back.

Here a support agent looks up an order before answering:

```python title="support.py"
from lllm import Prompt, Tactic, tool

ORDERS = {"A-1001": {"status": "shipped", "eta": "Thursday"}}


@tool(description="Look up an order by its number")
def lookup_order(order_id: str) -> dict:
    return ORDERS[order_id]


ANSWER = Prompt(
    path="support/answer",
    prompt="Answer the customer: {question}",
    function_list=[lookup_order],  # (1)!
)

agent = Tactic.quick(system_prompt="You are a support agent.", model="gpt-4o-mini")
agent.open("ticket")
agent.receive_prompt(ANSWER, {"question": "Where is my order A-1001?"})
session = agent.respond(return_session=True)

print(session.delivery.content)
#> Your order A-1001 has shipped and should arrive on Thursday.
print([call.name for calls in session.interrupts.values() for call in calls])  # (2)!
#> ['lookup_order']
```

1. The prompt offers the tool for this turn. Tools can also be attached to
   the agent itself; see [Giving tools to agents](#giving-tools-to-agents).
2. `interrupts` records every tool call the agent ran during `respond()`.

## Defining tools

Decorate a function with `@tool`. The description tells the model when to
use it, and each annotated parameter becomes a typed field:

```python title="define.py"
from typing import Literal

from pydantic import BaseModel

from lllm import tool


class Address(BaseModel):
    city: str
    country: str


@tool(
    description="Ship a package",
    prop_desc={"speed": "Delivery speed; express costs more"},  # (1)!
)
def ship(address: Address, speed: Literal["standard", "express"] = "standard") -> str:
    return f"{speed} shipping to {address.city}"


print(ship.required)
#> ['address']
print(ship.properties["speed"])
"""
{
    'enum': ['standard', 'express'],
    'type': 'string',
    'description': 'Delivery speed; express costs more (default: standard)',
}
"""
```

1. `prop_desc` documents parameters; defaults are added to the description.

The schema follows Pydantic: `list[str]` is an array of strings, `Literal` an
enum, a `BaseModel` a nested object, `Optional` a nullable value. Arguments
are validated before your function runs. The description defaults to the
docstring and the name to the function's name.

`strict` (OpenAI-style strict schemas) is chosen for you: it is on when every
parameter is required and flat, which is what strict mode accepts, and off
otherwise. Pass `strict=True` or `False` to decide yourself.

### Errors

An exception inside a tool does not crash the agent. The model receives
`Error: <message>` as the result and can try again or explain the problem:

```python title="tool_errors.py"
from lllm import FunctionCall, tool


@tool(description="Look up an order by its number")
def lookup_order(order_id: str) -> dict:
    raise KeyError(f"No order {order_id}")


call = lookup_order(FunctionCall(name="lookup_order", arguments={"order_id": "B-9"}))
print(call.result_str)
#> Error: 'No order B-9'
```

## Giving tools to agents

A tool can be offered in three places, from narrowest to broadest:

| Where | Offered | Use it for |
| --- | --- | --- |
| A prompt's `function_list` | while that prompt drives the turn | tools one step needs |
| The agent's system prompt | on every turn | tools an agent always has |
| The agent's config (`tools:`) | on every turn | registered tools, [proxies](proxies.md), and skills |

A prompt can turn the agent's own tools off for one turn with
`agent_tools=False`, for example a final summary that must not call anything.

In a project, tools live in files that `lllm.toml` discovers, and configs
refer to them by name:

```yaml
agent_configs:
  - name: analyst
    system_prompt: You analyze markets.
    tools: [search, my_pkg.proxies:fred]   # registered tools and proxies
```

## How a tool call runs

1. The model replies with one or more tool calls.
2. The agent runs each one and adds the results to the dialog.
3. The model sees the results and either calls more tools or answers.

The agent allows `max_interrupt_steps` rounds (5 by default). After the last
one, it asks the model to answer without tools. An identical call repeated in
the same turn gets a reminder to check the earlier result instead of running
again.

By default a result reaches the model framed with the call that produced it.
Pass `processor=` to `@tool` to control the text exactly:

```python title="processor.py"
from lllm import FunctionCall, tool


def as_price(result, function_call) -> str:
    return f"{function_call.arguments['ticker']}: ${result:.2f}"


@tool(description="Get a stock price", processor=as_price)
def price(ticker: str) -> float:
    return 123.456


print(price(FunctionCall(name="price", arguments={"ticker": "ACME"})).result_str)
#> ACME: $123.46
```

## Tactics as tools

A whole tactic can be a tool. The model calls it like any function, and the
tactic runs its own agents and prompts behind it. This makes LLLM programs
recursive: a single turn can hand work to a complete subsystem.

```python title="tactic_tool.py"
from lllm import Prompt, Tactic
from lllm.runtimes.native import tactic_as_function


class FactCheck(Tactic):
    """Check whether a claim is supported by our sources."""

    name = "fact_check"

    def call(self, claim: str) -> str:
        return "supported" if "paris" in claim.lower() else "unsupported"


ANSWER = Prompt(
    path="writer/answer",
    prompt="Answer, and check your main claim first: {question}",
    function_list=[tactic_as_function(FactCheck())],  # (1)!
)

agent = Tactic.quick(system_prompt="You write careful answers.", model="gpt-4o-mini")
agent.open("answer")
agent.receive_prompt(ANSWER, {"question": "What is the capital of France?"})
print(agent.respond().content)
#> Paris is the capital of France (checked: supported).
```

1. The tool's name, description, and parameters come from the tactic: its
   `name`, its docstring, and its input schema. Pass
   `parameter_mode="kwargs"` to expose an input model's fields as separate
   parameters.

Inside a tactic, the tool's calls are recorded in the caller's session. For
Pydantic AI agents, use `tactic.as_tool()` instead ([Pydantic AI](pydantic-ai.md)).

## More kinds of tools

- **Declarations.** A `Function` with only a schema can stand in a prompt
  while the implementation lives elsewhere; LLLM binds it by name when the
  prompt runs.
- **MCP servers.** `Prompt(mcp_servers_list=[MCP(server_label=..., server_url=...)])`
  offers a remote MCP server's tools, for providers that support it.
- **Provider tools.** Built-in provider tools, such as web search, go in
  `addon_args`.
- **APIs and code.** A [proxy](proxies.md) gives an agent a whole API it can
  read about and call, and the [sandbox](sandbox.md) lets it write and run
  Python against that API.
