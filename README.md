# Low-Level Language Models (LLLM)

<p align="center">
  <img src="assets/lllm-logo-text-dark.png" alt="LLLM" width="420">
</p>

LLLM is a lightweight framework for building advanced agentic systems: the
kind that decompose hard questions, call many tools, write and run code, and
need to be inspected turn by turn. It is *low-level* on purpose: instead of
hiding the agent loop, it gives you small, plain abstractions for every part
of it, so research and production systems can be customised without fighting
the framework.

Docs: [lllm.one](https://lllm.one) · Packages: [lllm.one/hub](https://lllm.one/hub/)

```bash
pip install "lllm-core[native]"
```

```python
from lllm import Tactic

print(Tactic.quick("What is the capital of France?").content)
```

## Design

An agentic system is a **program**:

- **Tactic is the program.** It owns a typed task interface and orchestrates
  agents; tactics compose, and a whole tactic can be handed to an agent as a
  tool.
- **Agent is the caller.** A model plus a call loop that handles retries,
  parse failures, and tool calls.
- **Prompt is the function.** A template with a parser, tools, and
  exception/interrupt handlers: the full specification of one turn.
- **Dialog is the mental state.** Each agent's own history, as a tree you can
  fork, replay, and inspect.

Configuration is declaration (prompts, tools, proxies, and agents are
discovered from `lllm.toml` and YAML configs and referenced by key), proxies
give agents an API surface they can program against (`run_python`,
`CALL_API`), and everything is typed with Pydantic.

```python
from pydantic import BaseModel
from lllm import Prompt, Tactic

TASK = Prompt(path="brief/task", prompt="Write three bullets about {topic}.")


class Topic(BaseModel):
    topic: str


class BriefTactic(Tactic):
    name = "brief"
    agent_group = ["writer", "editor"]
    input_model = Topic

    def call(self, task: Topic) -> str:
        writer, editor = self.agents["writer"], self.agents["editor"]
        writer.open("draft")
        writer.receive_prompt(TASK, {"topic": task.topic})
        editor.open("edit")
        editor.receive(f"Tighten this:\n{writer.respond().content}")
        return editor.respond().content


brief = BriefTactic({
    "global": {"model_name": "gpt-4o-mini"},
    "agent_configs": [
        {"name": "writer", "system_prompt": "You write crisp technical briefs."},
        {"name": "editor", "system_prompt": "You cut every unnecessary word."},
    ],
})
print(brief({"topic": "KV caches"}))
```

## One boundary for every runtime

`Tactic` is also a typed service boundary. Native programs, protocol-style
tactics (`_run`), Pydantic AI agents (`PydanticAITactic`), and plain functions
(`lllm.as_tactic`) are all validated, streamed, and served the same way:

```bash
lllm serve tactics:brief --port 8000     # POST /run, POST /stream, GET /info
```

`RemoteTactic` calls a served tactic as if it were local, `lllm://` refs and
`lllm.Resolver` wire systems together by name, and native agents can send
[inference hints](https://lllm.one/core/inference-hints/) (session identity
from dialogs, priorities, expected output length) to KV-cache-aware serving
stacks such as NVIDIA Dynamo, vLLM, and SGLang.

## Packages and the hub

```bash
lllm create my-app        # lllm.toml, prompts/, configs/, tactics/, main.py, tests
cd my-app
lllm hub validate .
lllm serve .              # serve the package's primary tactic
lllm hub publish . --local
```

One `lllm.toml` tells the native runtime what to discover and tells the hub
what the package is. Browse and download packages at
[lllm.one/hub](https://lllm.one/hub/).

## Modules

| Module | | Status |
| --- | --- | --- |
| `lllm` | Tactics, the native runtime (agents, prompts, dialogs, proxies, sandbox), runtimes, services | Stable |
| `lllm.hub` | The `lllm.toml` package format, validation, local hub, cards, static catalogs | Stable |
| `lllm.network` (SSSN) | Channels, events, artifacts, snapshots; `TacticProcessor` runs tactics over channels | Experimental |
| `lllm.shell` (AAAX) | Strategies that mount packages, tactics, and channels into one served API | Experimental |

One command covers them all: `lllm create | init | setup | inspect | serve | hub | network | shell`.

## Install options

| Extra | Adds |
| --- | --- |
| `native` | LiteLLM and YAML configs for native agents |
| `server` | FastAPI and Uvicorn |
| `client` | `httpx` for remote tactics and stores |
| `pydantic-ai` | the Pydantic AI adapter |
| `sandbox` | the Jupyter sandbox |
| `secure` | OS keyring for `lllm setup` |
| `all` | everything |

For development: `pip install -e ".[dev,docs]"` and `pytest`.

## Where it comes from

LLLM's tactic design grew out of the reusable reasoning units of Analytica,
which decomposes complex questions into grounded subpropositions, runs
specialised agents over them, and synthesises a robust analysis; it has since
served autonomous software development in Apeiron.

- Cheng, Junyan, Kyle Richardson, and Peter Chin. "Analytica: Soft
  Propositional Reasoning for Robust and Scalable LLM-Driven Analysis."
  *ICLR*, 2026.
- Cheng, Junyan, Ankit Srivastava, Jessie Zeng, Milenko Drinic, and Jack W.
  Stokes. "Apeiron: A Scalable LLM-agentic Framework for Autonomous
  Full-lifecycle Demand-optimized Application Synthesis." *Findings of ACL*,
  2026, pp. 3868-3899.

## License

Apache-2.0
