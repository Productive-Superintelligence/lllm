# Low-Level Language Models (LLLM)

<p class="psi-brand">
  <img src="assets/lllm-logo-text-dark.png#only-light" alt="LLLM" />
  <img src="assets/lllm-logo-text-white.png#only-dark" alt="LLLM" />
</p>

[lllm.one](https://lllm.one){ .psi-domain }

LLLM is a lightweight framework for building advanced agentic systems: the
kind that decompose hard questions, call many tools, write and run code, and
need to be inspected turn by turn. It is *low-level* on purpose. Instead of
hiding the agent loop, LLLM gives you small, plain abstractions for every part
of it (prompts, parsers, tool calls, exception and interrupt handling, dialog
state) so research and production systems can be customised without fighting
the framework.

```bash
pip install "lllm-core[native]"
```

```python
from lllm import Tactic

print(Tactic.quick("What is the capital of France?").content)
```

## Design Ideas

LLLM treats an agentic system as a **program**. Four abstractions carry the
whole design:

<div class="psi-tiles">
  <div class="psi-tile">
    <strong>Tactic is the program</strong>
    A typed unit of work that owns a task interface and wires agents together.
    Tactics compose, and a tactic can be handed to another agent as a tool.
  </div>
  <div class="psi-tile">
    <strong>Agent is the caller</strong>
    A model plus a call loop that handles retries, parse failures, and tool
    interrupts, so the program does not have to.
  </div>
  <div class="psi-tile">
    <strong>Prompt is the function</strong>
    A template with a parser, tools, and exception/interrupt handlers: the
    full specification of one agent turn.
  </div>
  <div class="psi-tile">
    <strong>Dialog is the mental state</strong>
    Each agent's own message history, kept as a tree you can fork, replay,
    and inspect.
  </div>
</div>

Around them run a few principles:

- **Configuration as declaration.** Prompts, tools, proxies, and tactics are
  discovered from `lllm.toml` and referenced by key, so a system's shape is
  declared rather than wired by hand.
- **Recursive by construction.** A whole tactic can be exposed as a
  prompt-callable tool, so a single agent turn can call a high-level
  subsystem.
- **Programs, not just tool calls.** Proxies give agents an API surface they
  can read and program against (`query_api_doc`, `run_python`, `CALL_API`),
  backed by a Jupyter sandbox for program synthesis.
- **Typed end to end with Pydantic.** Task inputs, structured outputs, and
  service contracts are Pydantic models.

The [architecture overview](core/architecture.md) walks through each idea with
code.

## The Native Runtime

The native runtime is LLLM's engine. A prompt, a tool, and a dialog:

```python
from lllm.runtimes.native import Dialog, FunctionCall, Prompt, Role, tool


@tool(description="Create a display label for a project area")
def label(topic: str) -> str:
    return topic.strip().title()


task = Prompt(
    path="planner/task",
    prompt="Plan the next checkpoint for {project}.",
    function_list=[label],
)

dialog = Dialog(owner="planner")
dialog.put_prompt(task, prompt_args={"project": "LLLM"}, name="operator")

call = task.get_function("label")(
    FunctionCall(name="label", arguments={"topic": "native core"})
)
dialog.put_text(call.result_str or "", role=Role.TOOL, name="label")

retry = dialog.fork(last_n=1)  # branch the agent's state and try again
print(dialog.overview())
```

Agents run prompts through this state with a call loop that retries, repairs
parse failures, and services tool calls. Start with the
[tutorials](tutorials/index.md) or the [agent reference](core/agent.md).

## Compatible With Pydantic

Pydantic is part of LLLM's contract, not an add-on:

- **Typed tasks and outputs.** A tactic declares its input and output as
  Pydantic models (`input_model`/`output_model`, or `input_type`/
  `output_type`). Inputs are validated on the way in and outputs on the way
  out, prompts can demand structured output with `format=Model`, and the JSON
  schemas are published with every service and tool.
- **Pydantic AI agents are tactics.** Wrap a `pydantic_ai.Agent` with
  `PydanticAITactic` and it can be served, packaged, and composed like any
  other tactic, while Pydantic AI keeps owning its tools, models, and output
  validation.
- **Tactics are Pydantic AI tools.** `tactic_as_tool` hands a tactic, such as a
  native multi-agent tactic, to a Pydantic AI agent as a typed tool.

See [Runtimes](protocol/runtimes.md) and the
[Pydantic AI tutorial](tutorials/10-pydantic-ai.md).

## One Boundary For Every Runtime

`Tactic` is both the program and the boundary. A tactic written with
`call()` and agents, a `_run()` hook, a Pydantic AI agent, or a plain Python
function is called, streamed, served, and packaged the same way.

```python
from pydantic import BaseModel
from lllm import Tactic


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    text: str


class Shout(Tactic[Question, Answer]):
    name = "shout"
    input_type = Question
    output_type = Answer

    def _run(self, input_value, *, context=None):
        return Answer(text=input_value.text.upper())


assert Shout().run({"text": "hello"}).text == "HELLO"
```

```bash
lllm serve tactics:Shout --port 8000      # FastAPI: /run, /stream, /info
```

Read more in [The Tactic](protocol/tactic.md) and [Services](protocol/services.md).

## Beyond One Tactic

<div class="psi-tiles">
  <div class="psi-tile">
    <strong>Packages and the hub</strong>
    One <code>lllm.toml</code> describes a package for the native runtime and the
    hub. Validate, publish, and browse packages at
    <a href="https://lllm.one/hub/">lllm.one/hub</a>.
  </div>
  <div class="psi-tile">
    <strong>SSSN: the network module</strong>
    Channels, events, artifacts, and snapshots that connect systems, plus a
    <code>TacticProcessor</code> that runs tactics over a channel.
  </div>
  <div class="psi-tile">
    <strong>AAAX: the shell module</strong>
    Mount packages, tactics, and channels into one strategy and serve it as a
    single API.
  </div>
  <div class="psi-tile">
    <strong>One CLI</strong>
    <code>lllm create</code>, <code>init</code>, <code>serve</code>,
    <code>hub</code>, <code>network</code>, and <code>shell</code>.
  </div>
</div>

SSSN and AAAX are experimental modules: [SSSN](sssn/index.md),
[AAAX](aaax/index.md), [packages](packages/manifest.md), [hub](packages/hub.md),
[CLI](reference/cli.md).

## Where It Comes From

LLLM's tactic design grew out of the reusable reasoning units needed by
**Analytica**, which decomposes complex societal, economic, political, and
scientific questions into grounded subpropositions, runs specialised agents
over them, and synthesises a more robust analysis. The same small boundary
has since served autonomous software development in **Apeiron**: demand
modeling, computer-use evaluation, activity tracing, and full-lifecycle
application synthesis.

- Cheng, Junyan, Kyle Richardson, and Peter Chin. "Analytica: Soft
  Propositional Reasoning for Robust and Scalable LLM-Driven Analysis."
  *ICLR*, 2026.
- Cheng, Junyan, Ankit Srivastava, Jessie Zeng, Milenko Drinic, and Jack W.
  Stokes. "Apeiron: A Scalable LLM-agentic Framework for Autonomous
  Full-lifecycle Demand-optimized Application Synthesis." *Findings of ACL*,
  2026, pp. 3868-3899.

## Next Steps

- [Getting Started](getting-started.md): install, first tactic, first agent.
- [Tutorials](tutorials/index.md): a guided path from agents to packages.
- [Architecture](core/architecture.md): the native design in depth.
