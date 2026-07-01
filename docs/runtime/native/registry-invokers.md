# Registry And Invokers

The native runtime has two important infrastructure pieces: the registry that
loads resources, and the invoker boundary that calls model providers.

## Runtime Registry

The native `Runtime` is a registry of `ResourceNode` objects keyed by qualified
refs:

```text
<package>.<section>:<resource_path>
```

Examples:

```text
demo.prompts:writer/system
demo.tools:search
demo.tactics:brief
demo.proxies:market_data
demo.configs:writer
demo.assets:logo.png
```

Typed helpers keep common access readable:

```python
from lllm.runtimes.native import Prompt, Runtime, tool
from lllm.runtimes.native.core import load_prompt, load_tool

runtime = Runtime()

prompt = Prompt(path="writer/system", prompt="You are concise.")
runtime.register_prompt(prompt, namespace="demo.prompts")


@tool(description="Echo text.")
def echo(text: str) -> str:
    return text


runtime.register_tool("echo", echo, namespace="demo.tools")

assert load_prompt("demo.prompts:writer/system", runtime=runtime).path
assert load_tool("demo.tools:echo", runtime=runtime).name == "echo"
```

## Package Discovery

Native package discovery reads `lllm.toml`, not `psi.toml`:

```toml
[package]
name = "demo"
version = "0.1.0"
description = "Native runtime demo."

[prompts]
paths = ["prompts"]

[tools]
paths = ["tools"]

[tactics]
paths = ["tactics"]

[dependencies]
packages = ["../shared as shared"]
```

`load_runtime()` searches for `lllm.toml`, `.lllm.toml`, or `LLLM.toml`, loads
dependencies, registers resources, and can auto-discover standard folders when
no config file exists.

## Resource Categories

| Category | Resource Types |
| --- | --- |
| Platform | `tactic`, `service`, `config`, `asset` |
| Native | `prompt`, `tool`, `proxy`, `context_manager` |
| Custom | User-defined sections discovered from `lllm.toml` |

Bare keys resolve through the runtime default namespace. Full keys always work
and are preferred in docs and packages because they are unambiguous.

## Invoker Boundary

`BaseInvoker.call()` is the provider boundary:

```python
invoke_result = invoker.call(
    dialog,
    model="gpt-4o-mini",
    model_args={"temperature": 0.2},
    parser_args={},
    responder="assistant",
)
```

It returns `InvokeResult`, which contains the raw provider response, actual
model args, parser or execution errors, the resulting native `Message`, and
cost through `invoke_result.cost`.

## LiteLLM

`LiteLLMInvoker` converts a native `Dialog` into provider-compatible messages.
It preserves assistant tool calls, tool messages, image messages, prompt
functions, MCP servers, structured output hints, logprobs, usage, cost, and
chat-vs-responses API differences.

```bash
python -m pip install -e ".[native]"
```

LiteLLM checks common provider environment variables at import time and raises
clear errors for partial Vertex AI, NVIDIA NIM, or Azure configuration. No
provider key is required for offline prompt, dialog, parser, registry, and
adapter tests.

## Streaming

Streaming uses `BaseStreamHandler`:

```python
from lllm.runtimes.native.invokers.base import BaseStreamHandler


class PrintStream(BaseStreamHandler):
    def handle_chunk(self, chunk_content: str, chunk_response):
        print(chunk_content, end="", flush=True)


agent.stream_handler = PrintStream()
message = agent.respond()
```
