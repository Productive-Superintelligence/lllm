# Dialogs

A dialog is one agent's mental state: every message it has seen and said, in
order. Dialogs are append-only. You add messages or fork a branch, but you
never edit history, so any point in a conversation can be replayed, compared,
or saved exactly as the agent saw it.

Each `receive()` and `respond()` adds to the active dialog, and `overview()`
prints it:

```python title="trip.py"
from lllm import Tactic

agent = Tactic.quick(system_prompt="You plan trips.", model="gpt-4o-mini")
agent.open("lisbon")
agent.receive("I have three days in Lisbon.")
agent.respond()
agent.receive("What should I pack for May?")
agent.respond()

print(agent.current_dialog.overview())
"""
[0. system (developer)]: You plan trips.

[1. user (user)]: I have three days in Lisbon.

[2. assistant (assistant)]: Spend a day each on Alfama, Belém, and Sintra.

[3. user (user)]: What should I pack for May?

[4. assistant (assistant)]: Light layers, walking shoes, and a thin jacket.
"""
```

Each entry shows the message's index, the sender's name, and its role. The
system prompt is sent with the `developer` role, so that is what it shows.
`overview()` cuts long messages at `max_length` characters (100 by default)
and shows an image by its type, size, and caption, such as
`<image/png, 2.0 KB> Revenue by quarter`, instead of its data.

## Messages

A dialog is a list of `Message` objects, in `dialog.messages`. `dialog.head`
is the first one (the system prompt) and `dialog.tail` the last one (usually
the latest reply). Every message has a role:

| Role | Added by | `name` |
| --- | --- | --- |
| `system` | `open()`, from the agent's system prompt | `system` |
| `user` | `receive()`, `receive_prompt()`, `receive_image()` | `user` |
| `assistant` | `respond()`: the final reply | the agent's name |
| `tool_call` | `respond()`: a reply that calls tools | the agent's name |
| `tool` | `respond()`: one tool's result | the tool's name |

[How LLLM works](concepts.md#one-turn-step-by-step) shows a turn with tool
calls. A reply carries more than its text:

```python title="reply.py"
from lllm import Tactic

agent = Tactic.quick(system_prompt="You name things.", model="gpt-4o-mini")
agent.open("names")
agent.receive("Suggest a name for a hiking app.")
reply = agent.respond(metadata={"step": "naming"})  # (1)!

print(reply.content)
#> Trailhead: short, outdoorsy, and hard to misspell.
print(reply.role.value, reply.model, reply.metadata["step"])
#> assistant gpt-4o-mini naming
print(reply.usage["prompt_tokens"], reply.usage["completion_tokens"])
#> 10 20
```

1. `metadata` is stored on the reply, next to the `dialog_id` that every
   message gets when it is added to a dialog.

| Field | Holds |
| --- | --- |
| `content` | The text; base64 data for an image |
| `parsed` | Structured output from the prompt's parser or [`format`](prompts.md#structured-output) |
| `function_calls` | The calls in a `tool_call` message: name, arguments, and result |
| `model`, `usage` | The model that wrote a reply, and its token counts and prices |
| `logprobs` | Token log probabilities, when the provider returns them |
| `metadata` | `dialog_id`, `tool_call_id` on tool results, `caption` on images, and your own keys |

Messages are Pydantic models: `message.to_dict()` and
`Message.from_dict(data)` round-trip every field.

## Usage and cost

`message.cost` turns a reply's usage into an `InvokeCost`: token counts
(prompt, completion, cached, reasoning) and dollar amounts from LiteLLM's
price table. `dialog.cost` adds up every message in the dialog:

```python title="cost.py"
from lllm import Tactic

agent = Tactic.quick(system_prompt="You plan trips.", model="gpt-4o-mini")
agent.open("lisbon")
agent.receive("I have three days in Lisbon.")
agent.respond()
agent.receive("What should I pack for May?")
agent.respond()

cost = agent.current_dialog.cost
print(cost.prompt_tokens, cost.completion_tokens, f"${cost.cost:.6f}")
#> 20 40 $0.000027
```

`InvokeCost` objects add with `+`, and `print(cost)` gives a readable
breakdown. The dialog counts replies that were kept; the
[call session](agents.md#responses) of a `respond()` also counts the replies
that were repaired, and a [tactic's session](tactics.md#sessions) adds up
every agent and sub-tactic call.

## Forking

To try several directions from the same point, fork the dialog. Each branch
starts with a copy of the parent's messages, and the parent is never
changed:

```python title="incident.py"
from lllm import Tactic

agent = Tactic.quick(
    system_prompt="You write incident updates.", model="gpt-4o-mini"
)
agent.open("incident")
agent.receive("Checkout broke for 12 minutes: a payment API key expired.")
agent.respond()

agent.fork("incident", "customers")  # (1)!
agent.receive("Write a one-line status update for customers.")
print(agent.respond().content)
#> Checkout was down 12 minutes; it works again.

agent.fork("incident", "engineers")  # (2)!
agent.receive("Write a one-line note for the on-call channel.")
print(agent.respond().content)
#> Expired payment API key broke checkout for 12 min. Add expiry alerts.

print({alias: len(d.messages) for alias, d in agent.dialogs.items()})
#> {'incident': 3, 'customers': 5, 'engineers': 5}

agent.fork("customers", "short", last_n=2)  # (3)!
print(agent.current_dialog.overview())
"""
[0. system (developer)]: You write incident updates.

[1. user (user)]: Write a one-line status update for customers.

[2. assistant (assistant)]: Checkout was down 12 minutes; it works again.
"""
```

1. `fork(alias, child_alias)` copies the `incident` dialog into a new one
   and switches to it.
2. The second branch starts from the same three messages. Neither branch
   sees the other, and `incident` still holds its three.
3. A fork can keep only part of the history, for example to retry from a
   clean point or to drop a long middle. This one keeps the system prompt
   and the last two messages.

| `fork()` argument | Default | Meaning |
| --- | --- | --- |
| `last_n` | `0` | Keep only the last `last_n` messages; `0` copies everything |
| `first_k` | `1` | With `last_n`, also keep the first `first_k` messages, usually the system prompt |
| `switch` | `True` | Make the new dialog the active one |

Each dialog records its place in a tree: `parent`, `children`, and `depth`,
plus a `tree_node` that holds only ids (its own `dialog_id`, its parent's,
and its children's) and the `split_point`, how many messages it copied.
`print(dialog.tree_overview())` draws the tree below a dialog.

!!! note "The agent loop works on a copy"
    For each model call, `respond()` works on a copy of the dialog. Repair
    messages and replies that failed to parse stay in that working copy, and
    only the accepted reply (with any tool calls and results) is added to
    your dialog. The copy is dropped when the call ends, so `children` and
    `tree_overview()` list only your own forks.

## Saving and loading

`to_dict()` returns plain JSON data, and `Dialog.from_dict()` rebuilds the
dialog from it:

```python title="save.py"
import json

from lllm import Dialog, Tactic

agent = Tactic.quick(system_prompt="You review code.", model="gpt-4o-mini")
agent.open("review")
agent.receive("Review this function name: getData2.")
agent.respond()
original = agent.current_dialog

with open("review.json", "w") as file:
    json.dump(original.to_dict(), file)

with open("review.json") as file:
    restored = Dialog.from_dict(json.load(file))
print(restored.overview())
"""
[0. system (developer)]: You review code.

[1. user (user)]: Review this function name: getData2.

[2. assistant (assistant)]: Too vague. Try fetch_invoices or load_users.
"""
print(restored.tail.model, restored.dialog_id == original.dialog_id)
#> gpt-4o-mini True
```

The data holds every message with its usage and metadata, the dialog's
owner, its tree node, the forks made from it (recursively), and the prompt
that drove the last turn. That prompt's tools are saved as schemas without
their code, and a Pydantic `format` as its import path and JSON schema.
Loading never imports code: a `format` model is used only if its module is
already imported, and its saved schema otherwise.

Use saved dialogs to review runs, compare branches offline, or build
datasets. To write a dialog by hand, for a test or a dataset, create a
`Dialog(owner=...)` and add messages with `put_text()`, `put_prompt()`, and
`put_image()`.

## Images

`receive_image()` adds an image to the dialog, with an optional caption. It
takes a file path, a Pillow image, or a base64-encoded PNG, JPEG, WebP, or
GIF:

```python title="chart.py"
from PIL import Image, ImageDraw

from lllm import Tactic

chart = Image.new("RGB", (160, 100), "white")  # (1)!
draw = ImageDraw.Draw(chart)
for i, height in enumerate([30, 45, 60, 85]):
    left = 15 + i * 35
    draw.rectangle([left, 100 - height, left + 25, 100], fill="navy")
chart.save("revenue.png")

agent = Tactic.quick(system_prompt="You read charts.", model="gpt-4o-mini")
agent.open("chart")
agent.receive_image("revenue.png", caption="Revenue by quarter, Q1 to Q4")
agent.receive("What is the trend in this chart?")
print(agent.respond().content)
#> Revenue rises every quarter, with the biggest jump in Q4.
```

1. Any image works; this one draws a small bar chart so the example runs on
   its own. Pillow comes with `lllm-core[native]`.

The model receives the caption as text followed by the image, labeled with
its format (`image/png` here). URLs and raw bytes are not accepted
(`ValueError`), and the model must support image input. An image does not
change which prompt drives the turn, so after `receive_prompt()`, that
prompt's parser and tools still apply.

## Long conversations

The dialog always keeps every message. When a conversation outgrows the
model's context window, give the agent a context manager: before each model
call, it trims the working copy that is sent, never the dialog itself.
`DefaultContextManager` keeps the system prompt and the most recent messages
that fit:

```python title="standups.py"
from lllm import Tactic
from lllm.runtimes.native import DefaultContextManager

agent = Tactic.quick(
    system_prompt="You track standup notes.", model="gpt-4o-mini"
)
agent.context_manager = DefaultContextManager(  # (1)!
    "gpt-4o-mini", max_tokens=240
)
agent.open("standups")
notes = {
    "Monday": "we started moving billing to the new API",
    "Tuesday": "staging now runs on the new API",
    "Wednesday": "load tests passed at twice normal traffic",
    "Thursday": "the old API became read-only",
    "Friday": "we switched production traffic over",
}
for day, note in notes.items():
    agent.receive(f"Standup note: on {day}, {note}.")
    agent.respond()

agent.receive("What did I say on Monday?")
print(agent.respond().content)  # (2)!
#> I no longer see Monday's note in our conversation.
print(len(agent.current_dialog.messages))
#> 13
```

1. `max_tokens` is the budget for what is sent, minus a margin kept free
   for the reply: 5,000 tokens, or half of a budget under 10,000. This tiny
   budget leaves 120 tokens: room for the last few notes.
2. The model saw the system prompt and the six most recent messages; the
   dialog still holds all 13.

Without `max_tokens`, the budget is the model's context window from
LiteLLM's model list (128,000 tokens for `gpt-4o-mini`). For a model
LiteLLM does not know, such as one on your own server, set `max_tokens`,
or register the model with `litellm.register_model`, giving its
`max_input_tokens`: without either, nothing is trimmed, and a warning says
so.

Messages are counted the way they are sent, tool calls included. An image
counts by its size, read from its header with Pillow, at OpenAI's
high-detail rate: 765 tokens for a 1024×768 photo, however long its data.
Everything else that is not text counts as 1,500 tokens (`IMAGE_TOKENS`):
an image whose size cannot be read (HEIC, data that does not decode, or
any image without Pillow), an image URL, or audio. Sizing these parts
never fetches anything. Text is counted by LiteLLM, and for Llama 2/3 and
Replicate models LiteLLM loads a tokenizer from Hugging Face; on a host
that cannot reach it, recent LiteLLM versions try again on every count,
and each try can take 20 seconds or more. Set
`litellm.disable_hf_tokenizer_download = True` to count with tiktoken
instead; `HF_HUB_OFFLINE=1` also stops the download, but LiteLLM then logs
a warning on every count.

The newest message is always sent whole, with a warning if it alone is
over the budget, and a tool call is kept or dropped together with its
results. An older message that only partly fits keeps its end, after an
`[...earlier content truncated...]` marker; an image that does not fit is
dropped whole, with a warning.

In a tactic, set a context manager per agent, or for every agent under
`global`:

```yaml
global:
  context_manager:
    max_tokens: 64000
agent_configs:
  - name: researcher
    system_prompt: You research topics.
  - name: archivist
    system_prompt: You keep every detail.
    context_manager:
      type: null        # never trim this agent's calls
```

Config builds a `DefaultContextManager` for each agent, so to change the
margin kept free for replies everywhere, set
`DefaultContextManager.SAFETY_BUFFER`; a value you set is used as is,
without the halving for small budgets.

For another policy, such as summarizing old turns, subclass `ContextManager`
(also from `lllm.runtimes.native`), set a `name`, and implement
`__call__(dialog)` to return the dialog to send, usually built with
`dialog.fork(...)`. Config builds it with `model_name` and `max_tokens`
keyword arguments, so accept both. Register the class on the runtime the
tactic uses, such as `get_default_runtime().register_context_manager(cls)`,
and select it with `type: <name>` ([Configuration](configuration.md)).

## Dialogs and agents

An agent keeps its dialogs under aliases: `agent.dialogs` maps each alias to
its `Dialog`, and `agent.current_dialog` is the active one.
[Agents](agents.md#conversations) covers `open()`, `switch()`, and `close()`;
`close(alias)` returns the dialog, for example to save it. `receive()`,
`receive_prompt()`, and `respond()` take `alias=` to work on another dialog
without switching to it.

A dialog belongs to the agent that opened it: its first message is that
agent's system prompt, and `owner` records the agent's name. To share work
between agents, pass content (`writer.receive(notes)`), not dialogs.

In a [tactic](tactics.md), agents and their dialogs last for one call. Read
them inside `call()`, or keep them with an [observer](observability.md),
which receives the call's agents when it finishes.
