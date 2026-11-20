# Typed I/O

A tactic declares what it accepts and what it returns as Pydantic types.
Every call validates both ends: `call()` works with real objects instead of
guessing at dicts, callers get a typed result or a clear error, and the same
types are published as JSON schemas that services and tools read.

Here a support tactic takes a `Ticket` and returns a `Triage`. The caller
passes a plain dict, and the model's JSON comes back as a validated object:

```python title="route.py"
from typing import Literal

from pydantic import BaseModel, Field

from lllm import Prompt, Tactic


class Ticket(BaseModel):
    subject: str
    body: str


class Triage(BaseModel):
    team: Literal["billing", "bugs", "accounts"]
    priority: int = Field(ge=1, le=3, description="1 is most urgent")


TRIAGE = Prompt(
    path="support/triage",
    prompt="Route this ticket.\nSubject: {subject}\n{body}",
    format=Triage,  # (1)!
)


class Route(Tactic):
    """Route a support ticket to a team."""

    name = "route"
    agent_group = ["router"]
    input_model = Ticket  # (2)!
    output_model = Triage

    def call(self, ticket: Ticket) -> Triage:
        router = self.agents["router"]
        router.open("triage")
        router.receive_prompt(TRIAGE, ticket.model_dump())  # (3)!
        return router.respond().parsed  # (4)!


config = {
    "global": {"model_name": "gpt-4o-mini"},
    "agent_configs": [
        {"name": "router", "system_prompt": "You route support tickets."}
    ],
}
route = Route(config)
triage = route({"subject": "Charged twice", "body": "I was billed twice."})
print(repr(triage))
#> Triage(team='billing', priority=2)
```

1. `format=` asks the model for JSON in the shape of `Triage`
   ([Structured output](prompts.md#structured-output)).
2. The tactic's contract. Every call validates its input into a `Ticket` and
   its result into a `Triage`, and `info()` publishes both.
3. The caller passed a dict, but `call()` receives a `Ticket`.
4. `parsed` is a dict of the reply's fields, already checked against
   `Triage` by `format=`. The tactic validates it into a `Triage` before
   the caller sees it.

## Declaring types

`input_model` and `output_model` take Pydantic models. They are aliases of
`input_type` and `output_type`, which take any type Pydantic can validate,
so use whichever reads better; a subclass's own declaration wins over the one
it inherits.

| Declared type | Accepts | Published schema |
| --- | --- | --- |
| A `BaseModel` | An instance, or data that validates into it, such as a dict | An object with the model's fields |
| `str`, `int`, `float`, `bool` | Values Pydantic converts, such as `"5"` for `int` | `string`, `integer`, `number`, `boolean` |
| `list[str]`, `dict[str, int]` | Lists and dicts of those | `array`, `object` |
| `Literal["a", "b"]` | One of the values | An `enum` |
| A dataclass or `TypedDict` | An instance or a dict | An object |

Typed I/O does not need agents. This tactic is plain Python:

```python title="stars.py"
from typing import Literal

from lllm import Tactic


class Stars(Tactic):
    """Turn a star rating into a label."""

    name = "stars"
    input_type = int
    output_type = Literal["negative", "neutral", "positive"]

    def call(self, stars: int) -> str:
        if stars >= 4:
            return "positive"
        return "neutral" if stars == 3 else "negative"


stars = Stars()
print(stars("5"))  # (1)!
#> positive
print(stars.info().input_schema)
#> {'type': 'integer'}
print(stars.info().output_schema)
#> {'enum': ['negative', 'neutral', 'positive'], 'type': 'string'}
```

1. The input is validated before `call()` runs, so `"5"` arrives as `5`.

Only these attributes are enforced and published. The type hints on `call()`
and the generic parameters in `Tactic[Ticket, Triage]` are for you and your
type checker; a tactic without declared types passes values through
unchecked, and its `info()` has no schemas.

## Validation

The input is checked before `call()` runs, so a bad request never reaches an
agent or costs a model call. The result is checked after `call()` returns,
so a bad result never reaches the caller:

```python title="validation.py"
from typing import Literal

from pydantic import BaseModel

from lllm import SchemaError, Tactic


class Ticket(BaseModel):
    subject: str
    body: str


class Triage(BaseModel):
    team: Literal["billing", "bugs", "accounts"]
    priority: int


class KeywordRoute(Tactic):
    """Route a ticket by keywords."""

    name = "keyword_route"
    input_model = Ticket
    output_model = Triage

    def call(self, ticket: Ticket) -> dict:
        if "charge" in ticket.body:
            return {"team": "billing", "priority": 2}
        return {"team": "sales", "priority": 3}  # (1)!


route = KeywordRoute()
print(repr(route({"subject": "Refund", "body": "Double charge."})))
#> Triage(team='billing', priority=2)

bad = [{"subject": "Refund"}, {"subject": "Hi", "body": "Pricing?"}]
for ticket in bad:
    try:
        route(ticket)
    except SchemaError as error:  # (2)!
        problem = error.__cause__.errors()[0]
        print(error.__cause__.title, problem["loc"], problem["msg"])
        #> Ticket ('body',) Field required
        #> Triage ('team',) Input should be 'billing', 'bugs' or 'accounts'
```

1. A bug: `sales` is not a team. The result is checked on the way out, so
   the caller gets an error instead of bad data.
2. The first ticket has no `body`. The second one passes, but its result
   names a team that does not exist.

Both failures raise `SchemaError` (from `lllm`), with a message like
`Value does not match schema Ticket: ...`. Its `__cause__` is Pydantic's
`ValidationError`, whose `errors()` list each problem with its location. The
call's [session](tactics.md#sessions) records the failure, and a
[service](serving.md) answers it with status 400.

## When the model's reply does not fit

Most results start as a model reply, and LLLM checks replies at two points:

| Check | Runs on | When it fails |
| --- | --- | --- |
| The prompt's `format=` or `parser=` | Each model reply | The error goes back to the model to fix; after `max_exception_retry` repairs (3), `AgentException` |
| The tactic's `output_model` | The tactic's result | `SchemaError`; the call fails |

`format=` validates every reply against its model, so a reply that does
not fit is repaired inside the agent loop instead of failing the call.
Providers with structured output enforce the schema as they generate, but
with other providers, or with validators a schema cannot express, a reply
can still be wrong:

```python title="repair.py"
from typing import Literal

from pydantic import BaseModel

from lllm import Prompt, Tactic


class Triage(BaseModel):
    team: Literal["billing", "bugs", "accounts"]
    priority: int


TRIAGE = Prompt(
    path="support/triage_checked",
    prompt=(
        "Route this ticket to billing, bugs, or accounts: {ticket}\n"
        'Reply with JSON like {{"team": "bugs", "priority": 2}}.'
    ),
    format=Triage,
)

agent = Tactic.quick(
    system_prompt="You route tickets.", model="gpt-4o-mini"
)
agent.open("triage")
ticket = "My invoice has the wrong company name."
agent.receive_prompt(TRIAGE, {"ticket": ticket})
session = agent.respond(return_session=True)

print(session.exception_retries_count)  # (1)!
#> 1
print(session.delivery.parsed)
#> {'team': 'billing', 'priority': 3}
```

1. The first reply named a team that does not exist. The model received
   `Error: The reply does not match Triage: team: Input should be
   'billing', 'bugs' or 'accounts'. Please fix.` and fixed it after one
   repair.

The repair message lists each problem with its field. Validators on the
model run too, so a rule the schema cannot express, such as one field
depending on another, is repaired the same way. For replies that are not
JSON, use a [parser](prompts.md#parsers), which raises `ParseError` to ask
for a repair. A prompt uses `format=` or a parser, not both: when
`format=` is set, the parser is not called.

`parsed` is the reply's JSON object as it came, which the model accepted,
so in a tactic with `output_model = Triage`, return it as is. The same
goes for steps inside `call()`:
`Plan.model_validate(planner.respond().parsed)` turns a planner's reply into
an object the rest of the program can use ([Tactics](tactics.md)). A
strict model (`ConfigDict(strict=True)`) rejects JSON forms such as a date
string in Python mode, so read `parsed` with
`Triage.model_validate_json(json.dumps(parsed))` and return the result.

## JSON schemas

`info()` returns the tactic's public description, a `TacticInfo`, with both
types as JSON schemas. They come from the types alone, so `call()` does not
run:

```python title="schemas.py"
from typing import Literal

from pydantic import BaseModel, Field

from lllm import Tactic


class Ticket(BaseModel):
    subject: str
    body: str


class Triage(BaseModel):
    team: Literal["billing", "bugs", "accounts"]
    priority: int = Field(ge=1, le=3)


class Route(Tactic):
    """Route a support ticket to a team."""

    name = "route"
    input_model = Ticket
    output_model = Triage

    def call(self, ticket: Ticket) -> Triage:
        return Triage(team="billing", priority=2)


info = Route().info()
print(info.name, "|", info.description)
#> route | Route a support ticket to a team.
print(info.input_schema["required"])
#> ['subject', 'body']
print(info.output_schema["properties"]["priority"])
#> {'maximum': 3, 'minimum': 1, 'title': 'Priority', 'type': 'integer'}
```

The rest of LLLM reads these schemas:

- **Services.** `lllm serve` validates each request's `input` against the
  input type, publishes both schemas at `GET /info` and in the OpenAPI docs,
  and answers a request that does not validate with status 400
  ([Serving](serving.md)).
- **Tools.** When a tactic becomes a tool, its input schema becomes the
  tool's parameters and its docstring the tool's description
  ([Tactics as tools](tools.md#tactics-as-tools)).
- **Inspection.** `lllm inspect module:tactic --json` prints the name,
  description, capabilities, and both schemas ([CLI](api/cli.md)).
- **Other runtimes.** Plain functions wrapped with `as_tactic()` (which
  reads the function's annotations) and [Pydantic AI](pydantic-ai.md)
  agents take `input_type` and `output_type` too, so every kind of tactic
  publishes the same contract ([Other runtimes](runtimes.md)).
