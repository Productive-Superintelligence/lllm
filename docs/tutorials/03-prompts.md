# 3. Prompts and Structured Output

**Goal:** write prompt templates, attach output parsers that turn replies into
structured data, watch the agent repair a malformed reply on its own, get
Pydantic-typed output, and reuse prompts through inheritance and a registry.

In LLLM a `Prompt` is more than a string. It is the complete behavior contract
for one agent turn:

1. **Template**: the text sent to the model, with `{variable}` placeholders.
2. **Output contract**: a `parser` (or a structured-output `format`) that turns
   the reply into data.
3. **Tools**: functions and MCP servers offered during the turn, on top of the
   agent's own tools ([Lesson 4](04-tools.md)).
4. **Handlers**: the messages used to recover from errors and to feed tool
   results back.

This is what makes LLLM agents reliable enough for program synthesis and other
research pipelines. When a reply breaks the output contract, the agent sends the
error back to the model and asks again, automatically, and only the corrected
reply is kept in the dialog.

Where Pydantic fits: `Prompt(format=MyModel)` asks the provider for JSON that
matches a Pydantic model, and a parser can validate replies with Pydantic so
that validation errors drive the same repair loop. Prompts and parsers are
Pydantic models themselves.

## Prerequisites

Finish [Lesson 2](02-agents-and-dialogs.md). The examples reuse its `agent`;
any agent works. Everything except the `respond()` calls runs locally, so you
can try prompts and parsers without an API key.

---

## Step 1: Create a prompt

```python
from lllm import Prompt

greet = Prompt(
    path="greet/system",
    prompt="You are a friendly assistant. Greet the user warmly.",
)

print(greet())   # render the template
```

`path` is the prompt's identifier. It shows up in logs and is the key under
which the prompt is registered (Step 11), so keep it unique.

---

## Step 2: Use template variables

Placeholders use Python `str.format()` syntax. Calling the prompt renders it:

```python
analyst = Prompt(
    path="analyst/system",
    prompt="You are an expert in {domain}. Today's date is {date}.",
)

print(analyst(domain="machine learning", date="2026-03-17"))
# You are an expert in machine learning. Today's date is 2026-03-17.

print(sorted(analyst.template_vars))                     # ['date', 'domain']
print(sorted(analyst.validate_args({"domain": "ml"})))   # ['date']: still missing
```

Rendering with a missing variable raises a `ValueError` that names the missing
variables, so a forgotten argument fails loudly instead of reaching the model.

Literal braces are always written doubled, whether or not the template has
variables:

```python
json_hint = Prompt(
    path="json/system",
    prompt='Reply with JSON like {{"answer": "..."}} about {topic}.',
)
print(json_hint(topic="cats"))   # Reply with JSON like {"answer": "..."} about cats.

plain = Prompt(path="json/plain", prompt='Reply with JSON like {{"answer": "..."}}.')
print(plain())                   # Reply with JSON like {"answer": "..."}.
```

---

## Step 3: Use prompts in a dialog

There are two places a prompt enters a dialog:

```python
agent.system_prompt = analyst
agent.open("session", prompt_args={"domain": "finance", "date": "2026-03-17"})
print(agent.current_dialog.head.content)
# You are an expert in finance. Today's date is 2026-03-17.
```

- `agent.open(alias, prompt_args=...)` renders the agent's system prompt with
  those arguments.
- `agent.receive_prompt(prompt, prompt_args)` renders any prompt and appends it
  as the next user message. You will use it in the next step.

This distinction matters because of one rule worth remembering:

> **The most recent prompt put into a dialog drives the next `respond()`.**
> Its parser, `format`, and handlers decide how the reply is checked and
> repaired, and any tools it declares are offered on top of the agent's own
> tools. `receive(text)` puts a plain prompt with no parser, so the reply comes
> back as free text.

So when a turn needs structured output, send it with `receive_prompt()`.

---

## Step 4: Parse XML tags

Attach a `DefaultTagParser` to the prompt for the turn:

```python
from lllm import Prompt
from lllm.runtimes.native import DefaultTagParser

extract = Prompt(
    path="extractor/task",
    prompt="""Extract the people and organizations mentioned in the text.
Answer in exactly this format:

<people>comma-separated names</people>
<organizations>comma-separated organization names</organizations>

Text: {text}""",
    parser=DefaultTagParser(xml_tags=["people", "organizations"]),
)
```

You can test the output contract without a model by parsing a sample reply:

```python
from pprint import pprint

pprint(
    extract.parse("<people>Alice, Bob</people><organizations>Acme Corp</organizations>"),
    sort_dicts=False,
)
```

```text
{'raw': '<people>Alice, Bob</people><organizations>Acme Corp</organizations>',
 'xml_tags': {'people': ['Alice, Bob'], 'organizations': ['Acme Corp']},
 'md_tags': {},
 'signal_tags': {}}
```

Each tag maps to a list because a tag can appear more than once. Now run it on
a real turn:

```python
agent.receive_prompt(extract, {"text": "Alice and Bob met the Acme Corp board in Paris."})
message = agent.respond()
print(message.parsed["xml_tags"]["people"])
```

`message.parsed` holds the same dictionary shape, built from the model's reply.

With this native parser, every tag listed in `xml_tags` (and in
`required_xml_tags`) must appear in the reply, otherwise parsing fails and the
repair loop in Step 7 kicks in. If a tag is optional, use the portable parser,
which only enforces the `required_*` lists:

```python
from lllm.parsers import DefaultTagParser as PortableTagParser

lenient = PortableTagParser(xml_tags=["organizations"], required_xml_tags=["people"])
print(lenient.parse("<people>Alice</people>")["xml_tags"])
# {'organizations': [], 'people': ['Alice']}
```

Native prompts accept either parser. Note that `from lllm import
DefaultTagParser` imports this portable parser; the native one lives in
`lllm.runtimes.native`.

---

## Step 5: Parse fenced code blocks

`md_tags` extracts fenced Markdown blocks by their language tag:

````python
coder = Prompt(
    path="coder/task",
    prompt="Write a Python function that {goal}. Put the code in a ```python block.",
    parser=DefaultTagParser(md_tags=["python"]),
)

parsed = coder.parse("Here it is:\n```python\ndef add(a, b):\n    return a + b\n```")
print(parsed["md_tags"]["python"])   # ['def add(a, b):\n    return a + b']
````

---

## Step 6: Detect signal tags

Signal tags let the model raise a flag. They are reported as booleans and never
fail parsing:

```python
review = Prompt(
    path="classifier/task",
    prompt="Is this review positive? If it is, include <POSITIVE> in your reply.\n\nReview: {review}",
    parser=DefaultTagParser(signal_tags=["POSITIVE"]),
)

print(review.parse("Clearly happy with the product. <POSITIVE>")["signal_tags"])  # {'POSITIVE': True}
print(review.parse("Mixed feelings.")["signal_tags"])                             # {'POSITIVE': False}
```

---

## Step 7: Watch the repair loop

When the parser raises, the agent does not fail right away. It shows the model
its own reply plus an error message and asks again, up to the agent's
`max_exception_retry` (default 3):

```python
strict = Prompt(
    path="math/answer",
    prompt="Compute {expression}. Put only the final number in <answer> tags.",
    parser=DefaultTagParser(xml_tags=["answer"]),
)

agent.receive_prompt(strict, {"expression": "17 * 23"})
session = agent.respond(return_session=True)

print(session.delivery.parsed["xml_tags"]["answer"])
print(session.exception_retries_count)   # how many repairs were needed
```

If a reply has no `<answer>` tag, the next request the model sees ends with its
bad reply followed by this message from the prompt's default handler:

```text
Error: Parsing errors:
No answer tags found, it should be provided as <answer>...</answer>. Please fix.
```

Repairs happen on a working copy of the dialog, so your dialog keeps only the
accepted reply. When retries run out, `respond()` raises `AgentException` with
the last parsing error. You can reword the repair message per prompt with
`handler=DefaultSimpleHandler(exception_msg="...")`;
[Lesson 8](08-advanced-patterns.md) shows custom handlers.

---

## Step 8: Write a custom parser

Subclass `DefaultTagParser` and override `parse()` to add your own validation.
Raise `ParseError` with a message the model can act on:

```python
from lllm.runtimes.native import DefaultTagParser, ParseError


def has_cycle(edges):
    graph = {}
    for src, dst in edges:
        graph.setdefault(src, []).append(dst)
    visiting, done = set(), set()

    def visit(node):
        if node in done:
            return False
        if node in visiting:
            return True
        visiting.add(node)
        if any(visit(nxt) for nxt in graph.get(node, [])):
            return True
        visiting.discard(node)
        done.add(node)
        return False

    return any(visit(node) for node in list(graph))


class DagParser(DefaultTagParser):
    """Parse 'a -> b' lines inside <edges> and reject cyclic graphs."""

    def parse(self, content: str, **kwargs) -> dict:
        parsed = super().parse(content, **kwargs)
        edges = [
            tuple(part.strip() for part in line.split("->"))
            for line in parsed["xml_tags"]["edges"][0].strip().splitlines()
        ]
        if has_cycle(edges):
            raise ParseError("The graph contains a cycle. Return an acyclic graph.")
        parsed["edges"] = edges
        return parsed


dag = DagParser(xml_tags=["edges"])
print(dag.parse("<edges>\nload -> clean\nclean -> train\n</edges>")["edges"])
# [('load', 'clean'), ('clean', 'train')]
```

Attach it with `Prompt(..., parser=DagParser(xml_tags=["edges"]))`. A cyclic
reply now goes back to the model with "The graph contains a cycle" until it
returns a valid graph. The native `ParseError` is a subclass of
`lllm.ParseError`, so `except lllm.ParseError` catches failures from either
parser.

A parser can also be a plain function `parser(content, **kwargs) -> dict`.
Whatever it returns is stored in `message.parsed`: a dict is kept as is
(non-dict values are wrapped as `{"value": ...}`), and `"raw"` is set to the
reply text unless the dict already has that key. Any exception it raises
triggers the repair loop. Extra keyword arguments come from
`agent.respond(parser_args={...})`.

---

## Step 9: Get Pydantic-typed output

There are two ways to get typed output.

### Validate with Pydantic inside a parser

This works with any model. Validation errors become repair messages, so the
model is told exactly which field is missing or malformed:

````python
from pydantic import BaseModel

from lllm import Prompt, find_md_blocks


class Summary(BaseModel):
    headline: str
    key_points: list[str]
    sentiment: str


def summary_parser(content: str, **kwargs) -> dict:
    blocks = find_md_blocks(content, "json")        # accept ```json fences too
    text = blocks[0] if blocks else content
    return Summary.model_validate_json(text).model_dump()


summarize = Prompt(
    path="summarizer/task",
    prompt=(
        "Summarize the article as a JSON object with the keys "
        "headline, key_points (a list of strings), and sentiment.\n\n{article}"
    ),
    parser=summary_parser,
)

article = "LLLM adds parsers and repair loops so agent outputs stay structured."
agent.receive_prompt(summarize, {"article": article})
summary = Summary.model_validate(agent.respond().parsed)
print(summary.headline, summary.key_points)
````

### Ask the provider for structured output

`format=` passes the model class to the provider's structured-output mode, so
the model is constrained to emit matching JSON:

```python
structured = Prompt(
    path="summarizer/structured",
    prompt="Summarize the following article:\n\n{article}",
    format=Summary,
)

agent.receive_prompt(structured, {"article": article})
message = agent.respond()
summary = Summary.model_validate(message.parsed)   # parsed holds the decoded JSON dict
```

Things to know about `format`:

- It needs a model and provider that support structured output, and the default
  completion API (`api_type="completion"`).
- `message.parsed` is the decoded JSON as a dict; call `Summary.model_validate()`
  to get the model instance.
- The prompt's `parser` is not run when `format` is set. Invalid JSON still
  triggers the repair loop.

---

## Step 10: Extend a prompt

`extend()` copies every field (template, parser, tools, handler, format) and
applies your overrides. A new `path` is required:

```python
base = Prompt(
    path="base/system",
    prompt="You are a helpful assistant working on {topic}.",
)

python_expert = base.extend(
    path="python_expert/system",
    prompt="You are a senior Python engineer working on {topic}. Always include type hints.",
)

print(python_expert(topic="a CLI tool"))
```

Calling `extend()` without `path` raises `ValueError`, because two prompts with
the same path would collide in the registry.

---

## Step 11: Register prompts and load them by path

Prompts can live in a registry and be fetched by path anywhere in your code:

```python
from lllm import Prompt, load_prompt, register_prompt

register_prompt(Prompt(
    path="greeter/system",
    prompt="You are a warm, friendly assistant. The user's name is {name}.",
))

greeter = load_prompt("greeter/system")
print(greeter(name="Ada"))
```

`register_prompt()` adds the prompt to the default runtime registry. Dialogs
accept registered paths directly, so `agent.receive_prompt("extractor/task",
{...})` works once `extractor/task` is registered.

In a project you rarely register prompts by hand: an LLLM package declares its
prompt folders in `lllm.toml` (a `prompts/` folder is picked up by default),
and every `Prompt` defined in those files is registered when the package loads.
`lllm create` sets this up for you ([Lesson 1](01-quick-start.md)), and
[Lesson 6](06-packages-and-hub.md) covers packages.

---

## Summary

| Feature | How |
|---|---|
| Template variables | `{variable}` in the template; literal braces always as `{{ }}` |
| Render manually | `prompt(var=value)` |
| Check variables | `prompt.template_vars`, `prompt.validate_args(args)` |
| Send a structured turn | `agent.receive_prompt(prompt, prompt_args)` |
| XML tags | `DefaultTagParser(xml_tags=[...])` from `lllm.runtimes.native` |
| Markdown blocks | `DefaultTagParser(md_tags=[...])` |
| Signal flags | `DefaultTagParser(signal_tags=[...])` |
| Optional tags | `lllm.parsers.DefaultTagParser(xml_tags=[...], required_xml_tags=[...])` |
| Custom validation | Subclass `DefaultTagParser` or pass a function; raise `ParseError` |
| Repair budget | `Agent(max_exception_retry=N)` |
| Pydantic output | Validate in a parser, or `Prompt(format=MyModel)` + `MyModel.model_validate(message.parsed)` |
| Inheritance | `prompt.extend(path=..., ...)` |
| Registry | `register_prompt(prompt)`, `load_prompt(path)` |

See the [Prompts reference](../core/prompts.md) for every field.

**Next:** [4. Tools, Proxies, and Sandbox](04-tools.md)
