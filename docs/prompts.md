# Prompts

A prompt is a function. Its template variables are the inputs, its parser or
`format` defines the output, and it carries everything else one turn needs:
the tools on offer and what to say when a reply comes back wrong. Because the
contract lives in the prompt, the agent can enforce it: a reply that does not
parse is sent back with the error until it does.

| Field | Role |
| --- | --- |
| `path` | A unique name, such as `support/classify`; prompts are registered and loaded by path |
| `prompt` | The template; `{name}` placeholders are its inputs |
| `parser` | Turns the reply into `parsed`; raising `ParseError` triggers a repair |
| `format` | A Pydantic model (or JSON schema) the reply must match |
| `function_list` | [Tools](tools.md) offered while this prompt drives the turn |
| `handler` | The messages sent after a parse error or a tool result |
| `agent_tools` | Whether the agent's own tools stay on offer (default `True`) |

Here is a prompt that classifies support tickets. Its output is whatever is
inside `<label>` tags:

```python title="classify.py"
from lllm import DefaultTagParser, Prompt, Tactic

CLASSIFY = Prompt(
    path="support/classify",
    prompt=(
        "Classify this ticket as billing, bug, or other.\n"
        "Ticket: {ticket}\n"
        "Answer inside <label></label> tags."
    ),
    parser=DefaultTagParser(required_xml_tags=["label"]),
)

agent = Tactic.quick(system_prompt="You triage support tickets.", model="gpt-4o-mini")
agent.open("triage")
agent.receive_prompt(CLASSIFY, {"ticket": "I was charged twice this month."})
reply = agent.respond()
print(reply.parsed["xml_tags"]["label"])
#> ['billing']
```

If the model forgets the tags, the parser raises, and the agent sends this
back to the model and asks again; your code only ever sees a reply that
parsed:

```python title="parse_error.py"
from lllm import DefaultTagParser, ParseError, Prompt

CLASSIFY = Prompt(
    path="support/classify",
    prompt="Classify: {ticket}. Answer inside <label></label> tags.",
    parser=DefaultTagParser(required_xml_tags=["label"]),
)

try:
    CLASSIFY.parse("billing")
except ParseError as error:
    print(error)
    """
    Parsing errors:
    Missing required XML tag: <label>...</label>
    """
```

## Templates

A template is a Python format string. Each `{name}` is an input, and doubled
braces are literal:

```python title="templates.py"
from lllm import Prompt

SUMMARIZE = Prompt(
    path="docs/summarize",
    prompt="Summarize {title} in {words} words. Return JSON like {{\"summary\": \"...\"}}.",
)
print(sorted(SUMMARIZE.template_vars))
#> ['title', 'words']
print(SUMMARIZE(title="the README", words=20))
#> Summarize the README in 20 words. Return JSON like {"summary": "..."}.
```

`agent.receive_prompt(prompt, args)` renders the prompt with `args` and adds
it to the dialog; a system prompt's variables are filled by
`agent.open(alias, prompt_args=...)`. A missing argument is an error before
any model call. To use another template language, give the prompt a
`renderer` (any object with a `render(template, **args)` method).

## Structured output

Give a prompt a Pydantic model as `format`, and the model is asked for JSON
matching it (OpenAI-style structured output where the provider supports
it). The reply is decoded into `parsed`, and invalid JSON is repaired like
any parse error:

```python title="structured.py"
from pydantic import BaseModel

from lllm import Prompt, Tactic


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str


EXTRACT = Prompt(
    path="finance/extract",
    prompt="Extract the invoice fields from:\n{text}",
    format=Invoice,
)

agent = Tactic.quick(system_prompt="You extract data.", model="gpt-4o-mini")
agent.open("extract")
agent.receive_prompt(EXTRACT, {"text": "ACME Corp. Amount due: 1,250.00 EUR"})
invoice = Invoice.model_validate(agent.respond().parsed)  # (1)!
print(invoice)
#> vendor='ACME Corp' total=1250.0 currency='EUR'
```

1. Inside a tactic you rarely do this by hand: declare `output_model =
   Invoice` and the tactic validates its result. See [Typed I/O](typed-io.md).

A `dict` JSON schema works as `format` too. For replies that are not JSON,
use a parser.

## Parsers

A parser is any object with a `parse(text, **kwargs)` method, or a plain
function. Whatever it returns becomes `parsed` (non-dict values are wrapped as
`{"value": ...}`), and whatever it raises is sent back to the model.

`DefaultTagParser` extracts tagged sections, which is often more robust than
JSON for long reasoning:

```python title="tags.py"
from lllm import DefaultTagParser

parser = DefaultTagParser(
    required_xml_tags=["reasoning", "answer"],
    md_tags=["python"],
    signal_tags=["DONE"],
)
fence = "`" * 3  # a Markdown code fence
parsed = parser.parse(
    "<reasoning>Sum the list.</reasoning>\n"
    "<answer>Use sum().</answer>\n"
    f"{fence}python\nprint(sum([1, 2, 3]))\n{fence}\n<DONE>"
)
print(parsed["xml_tags"])
#> {'reasoning': ['Sum the list.'], 'answer': ['Use sum().']}
print(parsed["md_tags"], parsed["signal_tags"])
#> {'python': ['print(sum([1, 2, 3]))']} {'DONE': True}
```

| Field | Extracts |
| --- | --- |
| `required_xml_tags` | `<tag>...</tag>` blocks; a missing tag is a parse error |
| `xml_tags` | `<tag>...</tag>` blocks, if present |
| `required_md_tags`, `md_tags` | fenced code blocks by language |
| `signal_tags` | `True` if `<TAG>` appears anywhere |

Write your own parser to enforce domain rules. The error message you raise is
exactly what the model reads, so make it actionable:

```python title="custom_parser.py"
from lllm import ParseError, Prompt


def percentage(text: str, **kwargs) -> float:
    try:
        value = float(text.strip().rstrip("%"))
    except ValueError:
        raise ParseError("Reply with a number only, such as 42 or 42%.")
    if not 0 <= value <= 100:
        raise ParseError(f"{value} is not a percentage; use 0 to 100.")
    return value


ESTIMATE = Prompt(
    path="forecast/estimate",
    prompt="What percentage of {group} use {tool}? Reply with a number.",
    parser=percentage,
)
print(ESTIMATE.parse("37%"))
#> {'value': 37.0, 'raw': '37%'}
```

Arguments passed to `agent.respond(parser_args={...})` reach the parser as
keyword arguments.

## Handlers

When a reply fails to parse, or tools return results, the agent asks the
prompt's handler what to send next. The default handler sends:

| Event | Message |
| --- | --- |
| Parse error | `Error: {error_message}. Please fix.` |
| Tool results | `{call_results}` |
| Tool budget used up | `You are reaching the limit of tool calls. Provide the final response.` (without tools) |

Change the wording without writing a class:

```python title="handler.py"
from lllm import DefaultTagParser, Prompt
from lllm.runtimes.native import DefaultSimpleHandler

STRICT = Prompt(
    path="support/classify_strict",
    prompt="Classify: {ticket}. Answer inside <label></label> tags.",
    parser=DefaultTagParser(required_xml_tags=["label"]),
    handler=DefaultSimpleHandler(
        exception_msg="Invalid reply: {error_message}\nAnswer again, tags only.",
    ),
)
print(STRICT.handler.exception_msg)
"""
Invalid reply: {error_message}
Answer again, tags only.
"""
```

For more control, subclass `BaseHandler` and implement `on_exception`,
`on_interrupt`, and `on_interrupt_final`. Each receives the prompt and the
current `AgentCallSession` and returns the next prompt, so a handler can, for
example, escalate its instructions on the third failure or ask another agent
how to fix the reply.

## Tools on a prompt

A prompt's `function_list` offers tools while it drives the turn, so each
step can bring exactly the tools it needs. The agent's own tools are offered
too, unless the prompt sets `agent_tools=False`:

```python test="skip" reason="fragment; see the Tools guide"
RESEARCH = Prompt(path="brief/research", prompt="Research {topic}.", function_list=[search])
SUMMARY = Prompt(path="brief/summary", prompt="Summarize the notes.", agent_tools=False)
```

See [Tools](tools.md).

## Reusing and composing prompts

`extend()` makes a variant that keeps every field except the ones you change,
and needs a new `path`:

```python title="extend.py"
from lllm import DefaultTagParser, Prompt

BASE = Prompt(
    path="analyst/base",
    prompt="Analyze {topic}. Answer inside <answer></answer> tags.",
    parser=DefaultTagParser(required_xml_tags=["answer"]),
)
FINANCE = BASE.extend(
    path="analyst/finance",
    prompt="As a financial analyst, analyze {topic}. Answer inside <answer></answer> tags.",
)
print(FINANCE.path, FINANCE.parser.required_xml_tags)  # the parser carries over
#> analyst/finance ['answer']
```

In a project, prompts live in Python files under a folder listed in
`lllm.toml`. LLLM discovers them and registers each under its package and
path, and agents and tactics refer to them by path (for example
`system_prompt_path: research/system`). Load one yourself with
`load_prompt("research/system")`. See [Configuration](configuration.md).
