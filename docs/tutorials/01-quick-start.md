# 1. Quick Start

**Goal:** talk to a model in one line, keep the conversation going with an
agent, point LLLM at a local model, and generate a project to grow into.

**You need:** `pip install "lllm-core[native]"` and either a provider key
(for example `OPENAI_API_KEY`) or a local OpenAI-compatible server.

## Step 1: one line

```python
from lllm import Tactic

response = Tactic.quick("What is the capital of France?")
print(response.content)  # Paris
```

`Tactic.quick()` is the zero-config entry point: it builds a prompt from the
default system prompt, creates one agent with a LiteLLM invoker, sends your
message, and returns the reply as a `Message`. `.content` is the text.

## Step 2: choose the model and the persona

```python
response = Tactic.quick(
    "What is 2 + 2?",
    system_prompt="You are a patient math teacher.",
    model="claude-opus-4-6",  # any LiteLLM model string
)
```

Any [LiteLLM](https://github.com/BerriAI/litellm) model identifier works:
`gpt-4o`, `claude-opus-4-6`, `mistral/mistral-large`, `gemini/...`, and so on,
with the provider's usual environment variable for the key.

## Step 3: keep the agent

Pass `return_agent=True` to get the agent back, or call `quick()` without a
query to get an agent you drive yourself:

```python
response, agent = Tactic.quick("Hi!", return_agent=True)
print(agent.name)  # assistant

agent = Tactic.quick(system_prompt="You are terse.")
agent.open("chat")                      # 1. open a dialog
agent.receive("Name a color.")          # 2. put a message in it
print(agent.respond().content)          # 3. run the agent's call loop
```

`open` → `receive` → `respond` is the pattern everything in LLLM builds on.
The dialog belongs to the agent: it is the agent's mental state, and
[Lesson 2](02-agents-and-dialogs.md) shows how to branch it, inspect it, and
run several of them side by side.

## Step 4: run against a local model

Extra keyword arguments to `quick()` become model arguments, so any
OpenAI-compatible server works, including vLLM, SGLang, and NVIDIA Dynamo:

```python
local = {
    "model": "hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct",
    "api_base": "http://127.0.0.1:8000/v1",
    "api_key": "local",
    "max_tokens": 64,
}
print(Tactic.quick("What is the capital of France? One word.", **local).content)
```

Self-hosted serving stacks can do more when they know what an agent is doing:
which calls belong to one reasoning chain, which dialogs were forked from
which, and how urgent a call is. Agents can send that as
[inference hints](../core/inference-hints.md) for KV-cache-aware routing and
priority scheduling.

## Step 5: generate a project

When the one-liner grows into a system, let LLLM lay out the project:

```bash
lllm create my-app            # native project, template "minimal"
cd my-app
pip install -e ".[dev]"
python main.py "Say hello in one sentence."
pytest                        # offline smoke test, no model calls
```

```text
my-app/
  lllm.toml            # the package: what to discover, and hub metadata
  prompts/assistant.py # Prompt objects, discovered by key
  configs/default.yaml # agents declared in YAML
  tactics/assistant.py # the program: a Tactic with agent_group
  main.py
  tests/test_smoke.py
```

Other templates: `--template pipeline` (planner, writer, reviewer),
`proxy` (an API surface for agents), `research` (batch experiments), and
`service` (served over HTTP). The tactic in the minimal template is a whole
program in a few lines:

```python
from lllm import Tactic


class AssistantTactic(Tactic):
    name = "assistant"
    agent_group = ["assistant"]

    def call(self, task: str) -> str:
        assistant = self.agents["assistant"]
        assistant.open("chat")
        assistant.receive(task)
        return assistant.respond().content
```

## Summary

| You want | Use |
| --- | --- |
| One question, one answer | `Tactic.quick("...")` |
| A conversation | `quick(return_agent=True)` or `quick()`, then open → receive → respond |
| A local or self-hosted model | `model=...`, `api_base=...` |
| A project to grow into | `lllm create my-app` |

**Next:** [2. Agents and Dialogs](02-agents-and-dialogs.md)
