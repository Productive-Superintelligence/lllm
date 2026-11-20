# Models and providers

Agents reach models through [LiteLLM](https://docs.litellm.ai/docs/providers),
so a model name is all it takes to switch providers, from OpenAI or
Anthropic to a vLLM server on your own machine. An agent never talks to a
provider itself: an *invoker* turns the dialog into a request and the
response into a message, so tools, parsing, repairs, and retries work the
same with every model.

```python title="providers.py"
from lllm import Tactic

for model in [
    "gpt-4o-mini",  # (1)!
    "anthropic/claude-sonnet-4-5",
    "gemini/gemini-2.5-flash",
    "ollama/llama3.2",  # (2)!
]:
    reply = Tactic.quick("Name one prime number above 90.", model=model)
    print(f"{model:30}{reply.content}")
    #> gpt-4o-mini                   97
    #> anthropic/claude-sonnet-4-5   97
    #> gemini/gemini-2.5-flash       97
    #> ollama/llama3.2               97
```

1. OpenAI models need no prefix. Other providers are chosen by the prefix
   before the `/`.
2. A model served by a local Ollama, with no key at all.

## Model names

A model name is LiteLLM's: a provider prefix and the provider's own name
for the model. Some common ones:

| Provider | Model name | Credentials |
| --- | --- | --- |
| OpenAI | `gpt-4o-mini`, `o4-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| Vertex AI | `vertex_ai/gemini-2.5-pro` | `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` |
| Azure OpenAI | `azure/<deployment>` | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` |
| NVIDIA NIM | `nvidia_nim/<model>` | `NVIDIA_NIM_API_KEY`, `NVIDIA_NIM_API_BASE` |
| OpenRouter | `openrouter/<model>` | `OPENROUTER_API_KEY` |
| Ollama | `ollama/llama3.2` | none |
| An OpenAI-compatible server | `openai/<model>` plus `api_base` | the server's key, if any |

[LiteLLM's provider list](https://docs.litellm.ai/docs/providers) has the
rest. In a tactic, set `model_name` once under `global` or per agent
([Configuration](configuration.md)).

## API keys

LiteLLM reads each provider's key from the environment variables in the
table above, such as `OPENAI_API_KEY`. LLLM checks the providers that need
several variables when it loads the invoker: if some of the Vertex AI, NVIDIA NIM, or Azure variables are set
but not all, it raises `ValueError` and lists them. If no known provider
key is set at all, it logs a warning, which you can ignore for a local
server. A key for one agent only, such as a local server's, can go in its
`model_args` as `api_key`. To store the keys a package needs, see
[Configuration](configuration.md#api-keys-and-environment-variables).

## Model settings

`model_args` are sent with every call. They come from the agent's config,
and `respond(args=...)` overrides them for one call. The call's session
shows what was sent:

```python title="model_args.py"
from lllm import Tactic

agent = Tactic.quick(
    system_prompt="You write haiku.",
    model="gpt-4o-mini",
    temperature=0.7,  # (1)!
    max_tokens=60,
)
agent.open("poem")
agent.receive("Write a haiku about autumn rain.")
session = agent.respond(args={"temperature": 1.0}, return_session=True)
print(session.invoke_traces[0][0].model_args)  # (2)!
"""
{'temperature': 1.0, 'max_tokens': 60, 'drop_params': True, 'stream': False}
"""
```

1. Extra keyword arguments to `Tactic.quick()` become `model_args`, except
   the agent's own settings, such as `max_llm_recall`
   ([below](#retries-and-rate-limits)).
2. `invoke_traces` keeps every model call of `respond()`, keyed by tool
   round. Each entry records the arguments that went to LiteLLM.

LLLM always sends `drop_params=True`, so a setting the provider does not
support, such as `seed` on a model without it, is dropped instead of
failing the call. Anything LiteLLM accepts can go in `model_args`:

| Setting | Does |
| --- | --- |
| `temperature`, `top_p`, `seed`, `stop` | Sampling and stop sequences |
| `max_tokens`, `max_completion_tokens` | Cap the reply's length |
| `api_base`, `api_key` | Where to send the request, and with which key |
| `timeout` | Seconds to wait for the provider |
| `reasoning_effort` | Reasoning budget, on models that have one |
| `logprobs`, `top_logprobs` | Token log probabilities on `message.logprobs` (prompts without `format`) |
| `extra_headers`, `extra_body` | Extra HTTP headers and request fields |

## Local and OpenAI-compatible servers

vLLM, SGLang, llama.cpp, and LM Studio speak the OpenAI API. Use the
`openai/` prefix with the model's name on the server, and point `api_base`
at it:

```python title="local_server.py"
from lllm import Tactic

reply = Tactic.quick(
    "Say hello in Spanish.",
    model="openai/Qwen/Qwen2.5-7B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="local",  # (1)!
)
print(reply.content)
#> ¡Hola!
print(reply.cost.cost)  # (2)!
#> 0.0
```

1. Any string works if the server does not check keys.
2. LiteLLM has no price for a model it does not know, so local calls cost
   nothing in LLLM's accounting. Token counts are still recorded.

In a config, the same settings go under `model_args`:

```yaml
global:
  model_name: openai/Qwen/Qwen2.5-7B-Instruct
  model_args:
    api_base: http://localhost:8000/v1
    api_key: local
```

LiteLLM also has `hosted_vllm/` for vLLM, and `ollama/` uses Ollama's
default address, `http://localhost:11434`, unless you set `api_base`.
Servers with KV-cache-aware routing, such as NVIDIA Dynamo, vLLM, and
SGLang, can also use what an agent knows about its dialogs; see
[Inference hints](inference-hints.md).

## Retries and rate limits

Providers fail now and then. The agent loop retries, with separate budgets
for each kind of failure ([The agent loop](agents.md#the-agent-loop)):

- **Rate limits** (`RateLimitError`, or HTTP 429) wait `min(60, 2**n)`
  seconds plus up to a second of jitter, up to `max_rate_limit_retries`
  times (8 by default).
- **Other provider errors**, such as a dropped connection, are retried
  after a second, up to `max_llm_recall` times (0 by default).
- **Replies that do not parse** are not provider errors. They are repaired
  through the prompt's handler, up to `max_exception_retry` times.

When a provider budget runs out, `respond()` raises the provider's
exception; when repairs run out, it raises `AgentException`. Here the first
request drops its connection:

```python title="recall.py"
from lllm import Tactic

agent = Tactic.quick(
    system_prompt="You answer briefly.", model="gpt-4o", max_llm_recall=2
)  # (1)!
agent.open("q")
agent.receive("What is the boiling point of water at sea level?")
session = agent.respond(return_session=True)
print(session.delivery.content)
#> 100 °C, or 212 °F.
print(session.llm_recalls_count, type(session.llm_recalls[0][0]).__name__)
#> 1 APIConnectionError
```

1. `Tactic.quick()` sets the agent's own settings, such as the retry
   budgets, by name. In a config, set `max_llm_recall: 2` on the agent or
   under `global`.

## Chat Completions or Responses

Agents use the Chat Completions API by default. Set `api_type: response`
on an agent to use OpenAI's Responses API instead; the agent loop is the
same, and only the request changes:

- A prompt's `format` is not supported (`ValueError`); use a parser.
- `Prompt(addon_args={"web_search": True})` adds the provider's web search
  tool for that turn, and `addon_args={"computer_use": {...}}` adds
  computer use.
- `max_output_tokens` (or `max_completion_tokens`) in `model_args` caps the
  reply, 32000 by default.
- Reasoning summaries, when the model returns them, are kept in
  `message.metadata["reasoning"]`.

## Cost

The invoker records each reply's token usage and its price from LiteLLM's
price table. `message.cost` is one reply's, and the totals add up: an
`AgentCallSession` has `cost`, a dialog has `cost`, and a tactic's call
session has `total_cost`, including nested tactic calls.

```python title="cost.py"
from lllm import Tactic

reply = Tactic.quick("Summarize Hamlet in one line.", model="gpt-4o-mini")
cost = reply.cost
print(cost.prompt_tokens, cost.completion_tokens, f"${cost.cost:.6f}")
#> 10 20 $0.000013
```

See [Observability](observability.md) for recording costs across runs.

## Invokers

The default invoker is `LiteLLMInvoker`. A tactic builds one for its agents
from the config's `invoker` key (`litellm` unless you name another), and
`build_invoker({"invoker": "litellm"})` builds one for an `Agent` you
construct yourself. One invoker serves every call of a tactic instance.

### Custom invokers

Write an invoker to reach a model LiteLLM does not cover. Subclass
`BaseInvoker` and implement `call()`, which receives the working dialog
and returns an `InvokeResult`:

```python title="custom_invoker.py"
from lllm import Message, Role, Tactic, register_invoker
from lllm.runtimes.native import BaseInvoker, InvokeResult


def house_model(text: str) -> str:
    """Stands in for the client of your own model server."""
    return f"[house-v1] {len(text.split())} words received."


class HouseInvoker(BaseInvoker):
    def call(self, dialog, model, model_args=None, parser_args=None,
             responder="assistant", metadata=None, **kwargs):
        content = house_model(dialog.tail.content)
        errors, parsed = [], {}
        try:
            parsed = dialog.top_prompt.parse(content, **(parser_args or {}))
        except Exception as error:
            errors.append(error)  # (1)!
        message = Message(
            role=Role.ASSISTANT, name=responder, content=content,
            parsed=parsed, model=model,
        )
        return InvokeResult(message=message, execution_errors=errors)


register_invoker("house", HouseInvoker)  # (2)!


class Count(Tactic):
    name = "count"
    agent_group = ["counter"]

    def call(self, text: str) -> str:
        counter = self.agents["counter"]
        counter.open("count")
        counter.receive(text)
        return counter.respond().content


count = Count(
    {
        "invoker": "house",  # (3)!
        "global": {"model_name": "house-v1"},
        "agent_configs": [
            {"name": "counter", "system_prompt": "You count words."}
        ],
    }
)
print(count("the quick brown fox"))
#> [house-v1] 4 words received.
```

1. A reply that fails to parse goes into `execution_errors` instead of
   raising, so the agent can send the prompt's repair message.
2. A builder receives the config's `invoker_config` (the whole config if
   there is none) and returns an invoker; the class itself is one.
   Registering a taken name raises unless you pass `overwrite=True`.
3. The config chooses the invoker. For an `Agent` you build yourself, pass
   `llm_invoker=HouseInvoker({})`.

A complete invoker also follows the rest of the contract:

- Read the turn's tools, `format`, and parser from `dialog.top_prompt`.
- Return a message with role `Role.TOOL_CALL` and `function_calls` when the
  model asks for tools; the agent runs them.
- Fill `message.usage` (`prompt_tokens`, `completion_tokens`, and
  `response_cost` when known) so costs add up.
- Call `stream_handler.handle_chunk()` with each piece of text, when one is
  given ([Streaming](streaming.md#streaming-tokens-from-an-agent)).
- Raise provider failures. The agent decides whether to retry.

To change how calls are made rather than where they go, such as caching or
routing, subclass `LiteLLMInvoker` from
`lllm.runtimes.native.invokers.litellm` and wrap its `call()`. For tests,
[`lllm.testing`](testing.md) answers model calls without writing an
invoker.
