# Testing

`lllm.testing` replaces the model, and only the model. Inside
`with ScriptedModel([...])` or `with FunctionModel(fn)`, every call a native
agent makes to a model is answered locally, while the rest of LLLM runs for
real: the dialog becomes messages, replies go through the prompt's parser
and `format`, bad replies are repaired, and tool calls run your functions.
A test therefore checks your program, not a mock of it, and needs no API
key or network.

```python title="moderate.py"
from lllm import Tactic
from lllm.testing import ScriptedModel


class Moderate(Tactic):
    name = "moderate"
    agent_group = ["moderator"]

    def call(self, comment: str) -> str:
        moderator = self.agents["moderator"]
        moderator.open("check")
        moderator.receive(f"Is this comment safe to post? {comment}")
        verdict = moderator.respond().content.lower()
        return "published" if verdict.startswith("yes") else "held"


moderate = Moderate({
    "global": {"model_name": "gpt-4o-mini"},
    "agent_configs": [
        {"name": "moderator", "system_prompt": "Answer yes or no."}
    ],
})

with ScriptedModel(["Yes."]):  # (1)!
    print(moderate("Great write-up, thanks!"))
    #> published

with ScriptedModel(["No, it is a personal attack."]) as model:
    print(moderate("You are an idiot."))
    #> held
print(model.requests[0].text)  # (2)!
#> Is this comment safe to post? You are an idiot.
```

1. Each block scripts the model's replies, so both branches of `call()` are
   tested.
2. Every request is kept, so a test can also check what the model was
   asked.

## Scripted replies

`ScriptedModel(replies)` answers calls in order, and once the replies run
out, the last one repeats. A reply is text, a `ToolCall`, or a list of
`ToolCall`s, which the model makes at once. Tool calls run your real tools:

```python title="scripted_tools.py"
from lllm import Prompt, Tactic, tool
from lllm.testing import ScriptedModel, ToolCall

STOCK = {"widget": 3}


@tool(description="Units in stock for a product")
def in_stock(product: str) -> int:
    return STOCK.get(product, 0)


ASK = Prompt(path="shop/ask", prompt="{question}", function_list=[in_stock])

agent = Tactic.quick(system_prompt="You run a shop.", model="gpt-4o-mini")
agent.open("chat")
agent.receive_prompt(ASK, {"question": "Do you have widgets?"})

replies = [ToolCall("in_stock", {"product": "widget"}), "Yes, 3 left."]
with ScriptedModel(replies) as model:
    session = agent.respond(return_session=True)

print(session.delivery.content)
#> Yes, 3 left.
call = session.tool_calls[0]
print(call.name, call.arguments, call.result)  # (1)!
#> in_stock {'product': 'widget'} 3
print([message.role.value for message in agent.current_dialog.messages])
#> ['system', 'user', 'tool_call', 'tool', 'assistant']
```

1. `in_stock` really ran: the result came from `STOCK`, and the second of
   the two `model.requests` carried it back to the model.

## Answering from the request

When the replies depend on what the model is asked, pass a function to
`FunctionModel`. It receives a `ModelRequest` and returns a reply, so one
function can serve a whole test suite:

```python title="function_model.py"
from lllm import Prompt, Tactic, tool
from lllm.testing import FunctionModel, ModelRequest, ToolCall


@tool(description="Today's high in a city, in °C")
def high(city: str) -> int:
    return {"Oslo": 4, "Rome": 18}[city]


def fake_model(request: ModelRequest):
    if request.tool_results():  # (1)!
        degrees = request.tool_results()[-1].split("---")[1].strip()
        return f"About {degrees}°C today."
    if "high" in request.tools:
        city = request.text.split()[-1].rstrip("?")
        return ToolCall("high", {"city": city})
    return "Ask me about the weather."


ASK = Prompt(path="weather/ask", prompt="{question}", function_list=[high])
agent = Tactic.quick(
    system_prompt="You report weather.", model="gpt-4o-mini"
)
agent.open("today")
agent.receive_prompt(ASK, {"question": "How warm is it in Rome?"})

with FunctionModel(fake_model) as model:
    print(agent.respond().content)
    #> About 18°C today.
print([request.tools for request in model.requests])
#> [['high'], ['high']]
```

1. `tool_results()` holds the tool messages since the last user message:
   none on the first call, the results after a tool round. Each result is
   framed with the call that produced it, the value between `---` lines.

| `ModelRequest` | Holds |
| --- | --- |
| `text` | The last message's content, as text |
| `system` | The system prompt |
| `tools` | Names of the tools on offer |
| `tool_results()` | Contents of the tool results since the last user message |
| `messages` | Every message, as the provider would receive it |
| `last_message` | The last message, as a dict |
| `response_format` | The response format the prompt asked for, if any |
| `model` | The model name |

## Testing repairs

Script a reply that fails the prompt's parser or `format`, then a good one.
The agent sends the error back, and your code only sees the reply that
parsed:

```python title="repairs.py"
from pydantic import BaseModel

from lllm import Prompt, Tactic
from lllm.testing import ScriptedModel


class Rating(BaseModel):
    stars: int


RATE = Prompt(
    path="review/rate", prompt="Rate 1-5: {review}", format=Rating
)

agent = Tactic.quick(system_prompt="You rate reviews.", model="gpt-4o-mini")
agent.open("rate")
agent.receive_prompt(RATE, {"review": "Fast shipping, crushed box."})

with ScriptedModel(["Four stars.", '{"stars": 4}']) as model:
    session = agent.respond(return_session=True)

print(session.delivery.parsed, session.exception_retries_count)
#> {'stars': 4} 1
print(model.requests[1].text)  # (1)!
#> Error: Expecting value: line 1 column 1 (char 0). Please fix.
```

1. The repair message the model received, from the prompt's
   [handler](prompts.md#handlers).

A script that never parses ends in `AgentException` (from
`lllm.runtimes.native`) after `max_exception_retry` repairs, which is how
to test that limit.

## Testing a tactic end to end

Run the tactic with `return_session=True` and check the record: which
agents ran, the tools they called, the repairs, and the result. A tactic's
agents and dialogs last for one call, so to check its dialogs, read them in
an observer. [Observability](observability.md) covers both. Here is a test
module for the tactic above, with pytest:

```python title="test_moderate.py" test="skip" reason="imports moderate.py"
import pytest

from lllm.testing import FunctionModel, ScriptedModel
from moderate import moderate


@pytest.fixture(autouse=True)
def no_real_model():  # (1)!
    def refuse(request):
        raise AssertionError(f"Unexpected model call: {request.text!r}")

    with FunctionModel(refuse):
        yield


@pytest.mark.parametrize(
    ("reply", "outcome"),
    [("Yes.", "published"), ("No.", "held"), ("I'm not sure.", "held")],
)
def test_verdicts(reply, outcome):
    with ScriptedModel([reply]):  # (2)!
        assert moderate("A comment.") == outcome


def test_one_agent_call_per_comment():
    with ScriptedModel(["Yes."]):
        session = moderate("Hello!", return_session=True)
    assert session.agent_call_count == 1
    assert session.agent_sessions["moderator"][0].state == "success"
```

1. A guard for the whole suite: any model call a test did not script fails
   it, instead of reaching a provider.
2. Fake models nest. The innermost one answers, and the outer one takes
   over again when the block ends.

## Good to know

- The fake models patch LiteLLM's completion call for the whole process,
  so they also answer calls from other threads, such as `bcall` workers.
  Do not run tests with different fake models in parallel threads.
- They answer chat-completion calls, the default. An agent configured with
  `api_type: response` still calls the provider.
- LiteLLM builds the responses, so the model name must be one it can route:
  any real name, or a provider prefix such as `openai/my-model`.
- Token counts are LiteLLM's placeholder usage, the same for every call, so
  costs in tests are stable but not realistic.
- Plain-Python tactics need no fake model at all, and
  [Pydantic AI](pydantic-ai.md) agents use Pydantic AI's own `TestModel`
  and `FunctionModel`.

Every example in these docs runs this way: a `FunctionModel` answers each
request by matching it against a list of rules
([`tests/docs_model.py`](https://github.com/Productive-Superintelligence/lllm/blob/main/tests/docs_model.py)),
and the docs tests check that each example prints what the page shows.
