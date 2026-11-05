# Getting Started

From install to a multi-agent program, a served API, and a package, in a few
minutes.

## Install

```bash
pip install "lllm-core[native,server]"
```

The distribution is `lllm-core`; the import is `lllm`.

| Extra | Adds |
| --- | --- |
| `native` | LiteLLM (any provider) and YAML configs for the native agent runtime |
| `server` | FastAPI and Uvicorn for `lllm serve` and the service apps |
| `client` | `httpx` for `RemoteTactic` and `RemoteStore` |
| `pydantic-ai` | the Pydantic AI runtime adapter |
| `sandbox` | the Jupyter sandbox for program synthesis |
| `secure` | OS keyring storage for `lllm setup` |
| `all` | everything above |

Set a provider key (for example `OPENAI_API_KEY`), or point LLLM at a local
OpenAI-compatible server as shown in [the quick start](tutorials/01-quick-start.md#step-4-run-against-a-local-model).

## 1. One line

```python
from lllm import Tactic

print(Tactic.quick("What is the capital of France?").content)
```

`Tactic.quick` builds one agent with no configuration. Keep it with
`return_agent=True` and continue the conversation with
`agent.receive(...)` and `agent.respond()`.

## 2. A program with agents

A tactic is the program of an agentic system. Declare the agents it needs in
`agent_group`; every call gets fresh agents built from the config, and a call
session records what they did and what it cost:

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
        draft = writer.respond().content
        editor.open("edit")
        editor.receive(f"Tighten this:\n{draft}")
        return editor.respond().content


config = {
    "global": {"model_name": "gpt-4o-mini"},
    "agent_configs": [
        {"name": "writer", "system_prompt": "You write crisp technical briefs."},
        {"name": "editor", "system_prompt": "You cut every unnecessary word."},
    ],
}

brief = BriefTactic(config)
session = brief({"topic": "KV caches"}, return_session=True)
print(session.delivery, session.agent_call_count, session.total_cost)
```

Each agent keeps its own dialog. Prompts are reusable specifications of a
turn; `receive_prompt` renders one into the dialog, and `respond()` runs the
agent's call loop (retries, parsing, tool calls).
[Architecture](core/architecture.md) explains the design.

## 3. Serve it

Every tactic is also a typed service boundary. `input_model` is validated on
the way in and published as a JSON schema:

```python
from lllm.services import create_tactic_app

app = create_tactic_app(brief)   # POST /run, POST /stream, GET /info
```

```bash
lllm serve tactics:brief --port 8000
curl -X POST http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{"input": {"topic": "KV caches"}}'
```

## 4. Make it a package

```bash
lllm create briefs        # lllm.toml, prompts/, configs/, tactics/, main.py, tests
cd briefs
lllm hub validate .
lllm serve .              # serves the package's primary tactic
```

The generated `lllm.toml` tells the native runtime what to discover (prompts,
YAML agent configs, tactics) and tells the hub what the package is. Publish it
with `lllm hub publish . --local`, or browse shared packages at
[lllm.one/hub](https://lllm.one/hub/).

## Where next

- [Tutorials](tutorials/index.md): eleven lessons from agents to channels.
- [Architecture](core/architecture.md): the native design and its ideas.
- [The Tactic](protocol/tactic.md): the one class every runtime shares.
- [Inference hints](core/inference-hints.md): infra-aware serving with
  Dynamo, vLLM, and SGLang.
