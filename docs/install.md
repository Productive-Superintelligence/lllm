# Installation

LLLM needs Python 3.10 or later. The distribution is called `lllm-core`; you
import it as `lllm`.

```bash
pip install "lllm-core[native]"
```

or with uv:

```bash
uv add "lllm-core[native]"
```

The base package has a single dependency (Pydantic) and gives you the typed
tactic boundary. The `native` extra adds what agents need to call models.

## Extras

| Extra | Adds | Install it for |
| --- | --- | --- |
| `native` | LiteLLM, YAML configs | agents calling models (most users) |
| `server` | FastAPI, Uvicorn | [serving](serving.md) tactics with `lllm serve` |
| `client` | httpx | calling served tactics with `RemoteTactic` |
| `sandbox` | Jupyter client | the [code sandbox](sandbox.md) |
| `pydantic-ai` | Pydantic AI | [running Pydantic AI agents](pydantic-ai.md) as tactics |
| `secure` | keyring | storing API keys with `lllm setup` |
| `all` | everything above | |

```bash
pip install "lllm-core[native,server]"
```

## Connect a model

Agents call models through [LiteLLM](https://docs.litellm.ai/docs/providers),
so any provider it supports works. Set the provider's key in the environment
and name the model:

```bash
export OPENAI_API_KEY=sk-...       # or ANTHROPIC_API_KEY, GEMINI_API_KEY, ...
```

```python test="skip" reason="calls a real provider"
from lllm import Tactic

print(Tactic.quick("Say hi", model="anthropic/claude-sonnet-4-5").content)
```

Model names follow LiteLLM: `gpt-4o-mini`, `anthropic/claude-sonnet-4-5`,
`gemini/gemini-2.5-flash`, `ollama/llama3.2`, and so on. In a tactic, set the
model once in the config (`"global": {"model_name": ...}`) or per agent.

### Local models

Point an agent at any OpenAI-compatible server (vLLM, SGLang, Ollama,
llama.cpp) with the `openai/` prefix and the server's address:

```python test="skip" reason="needs a running model server"
from lllm import Tactic

reply = Tactic.quick(
    "Say hi",
    model="openai/Qwen/Qwen2.5-7B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="local",
)
print(reply.content)
```

Extra keyword arguments to `Tactic.quick()` are sent with every model call;
in a config they go under `model_args`. [Models](models.md) covers providers,
settings, and custom invokers.

## Check the install

You can run an agent without any key: [`lllm.testing`](testing.md) answers
model calls locally.

```python title="check_install.py"
from lllm import Tactic
from lllm.testing import ScriptedModel

with ScriptedModel(["LLLM is installed."]):
    print(Tactic.quick("Are you there?").content)
    #> LLLM is installed.
```

## The command line

Installing LLLM adds the `lllm` command:

```bash
lllm --help
lllm create my-app      # start a project from a template
lllm serve my-app       # serve its tactic over HTTP
lllm hub validate my-app
```

See the [CLI reference](api/cli.md).

## From source

```bash
git clone https://github.com/Productive-Superintelligence/lllm
cd lllm
pip install -e ".[all,dev,docs]"
pytest
```
