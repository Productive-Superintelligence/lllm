# Invokers

Invokers are the execution engines of the native runtime. An invoker takes the
state of a `Dialog`, calls a model API, and turns the response into the next
`Message`. Keeping the invoker that small is what lets the agent loop stay
provider-agnostic: retries, parsing policy, and tool execution live in the
agent, while the invoker only talks to the provider.

The default invoker is built on [LiteLLM](https://github.com/BerriAI/litellm),
which reaches 100+ providers (OpenAI, Anthropic, Gemini, Vertex AI, Azure,
Bedrock, local models through Ollama or vLLM, and more) through one interface.

```bash
pip install "lllm-core[native]"
```

---

## The `BaseInvoker` interface

Every invoker subclasses `BaseInvoker` and implements one method, `call`:

```python
class BaseInvoker(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def call(
        self,
        dialog: Dialog,
        model: str,
        model_args: Optional[Dict[str, Any]] = None,
        parser_args: Optional[Dict[str, Any]] = None,
        responder: str = "assistant",
        metadata: Optional[Dict[str, Any]] = None,
        api_type: APITypes = APITypes.COMPLETION,
        stream_handler: Optional[BaseStreamHandler] = None,
    ) -> InvokeResult:
        ...
```

`BaseInvoker` and `BaseStreamHandler` live in
`lllm.runtimes.native.invokers.base`.

### Arguments

| Argument | Description |
| --- | --- |
| `dialog` | The working dialog. The invoker reads `dialog.top_prompt` for tools, MCP servers, `format`, and parser. |
| `model` | Model id, e.g. `"gpt-4o"` or `"anthropic/claude-sonnet-4-5"`. |
| `model_args` | Provider arguments (`temperature`, `max_tokens`, `api_base`, ...), already merged by the agent. |
| `parser_args` | Keyword arguments for the prompt's parser. |
| `responder` | Name of the agent replying; becomes the reply's `name`. |
| `metadata` | Tracking data attached to the reply's `metadata`; not sent to the model. |
| `api_type` | `APITypes.COMPLETION` (Chat Completions) or `APITypes.RESPONSE` (OpenAI Responses API). |
| `stream_handler` | Optional `BaseStreamHandler` for real-time chunks. |

### Return value: `InvokeResult`

```python
@dataclass
class InvokeResult:
    raw_response: Any = None                                         # raw provider response
    model_args: Dict[str, Any] = field(default_factory=dict)         # args actually sent
    execution_errors: List[Exception] = field(default_factory=list)  # parse/validation errors
    message: Optional[Message] = None                                # the conversational message
```

`InvokeResult` bundles the clean `Message` that goes into the dialog with the
diagnostics the loop needs. The agent uses `has_errors` to decide whether to
repair, and appends `message` to the dialog:

```python
invoke_result = invoker.call(dialog, model, model_args, ...)
if invoke_result.has_errors:
    raise AgentException(invoke_result.error_message)   # -> exception handler
dialog.append(invoke_result.message)
```

### The invoker contract

An invoker should:

- read tools and the output contract from `dialog.top_prompt`;
- return a message with role `ASSISTANT` for a final reply, or `TOOL_CALL` with
  `function_calls` when the model asks for tools;
- run the prompt's parser on final replies (`prompt.parse(content,
  **parser_args)`) and put any parser exception in `execution_errors` instead of
  raising, so the agent can send a repair prompt;
- fill `message.usage` (token counts, and `response_cost`, `prompt_cost`,
  `completion_cost` when known) so costs aggregate;
- raise on provider failures (network, auth, rate limits). The agent retries
  rate limits (an exception named `RateLimitError`, one with
  `status_code == 429`, or a known rate-limit message) with backoff, and treats
  other errors as LLM recalls.

---

## Streaming

Streaming uses a callback so dialog state stays clean and synchronous.
Subclass `BaseStreamHandler`:

```python
from lllm.runtimes.native.invokers.base import BaseStreamHandler


class ConsoleStreamer(BaseStreamHandler):
    def handle_chunk(self, chunk_content: str, chunk_response):
        print(chunk_content, end="", flush=True)


agent.stream_handler = ConsoleStreamer()              # used by every respond()
invoke_result = invoker.call(dialog, "gpt-4o", stream_handler=ConsoleStreamer())
```

The invoker streams text chunks to the handler as they arrive and still returns
a complete `InvokeResult` at the end. `chunk_response` is the raw chunk (a
Chat Completions chunk or a Responses API event), for anything beyond text.

---

## The LiteLLM invoker

```python
from lllm.runtimes.native.invokers import build_invoker

invoker = build_invoker({"invoker": "litellm"})
# equivalently:
from lllm.runtimes.native.invokers.litellm import LiteLLMInvoker
invoker = LiteLLMInvoker({})
```

A `Tactic` with `agent_group` builds this invoker for you unless its config names another
one.

### 1. Model names

LiteLLM routes by model prefix:

- **OpenAI:** `gpt-4o` or `openai/gpt-4o`
- **Anthropic:** `anthropic/claude-sonnet-4-5`
- **Google:** `gemini/gemini-2.5-pro`
- **Ollama (local):** `ollama/llama3`
- **Azure:** `azure/my-gpt4-deployment`

See the [LiteLLM providers](https://docs.litellm.ai/docs/providers) for every
supported model.

### 2. API keys

LiteLLM reads the standard environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
```

When the module is imported, LLLM checks provider groups that need several
variables and raises `ValueError` for a partial setup: Vertex AI
(`VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`), NVIDIA NIM (`NVIDIA_NIM_API_KEY`,
`NVIDIA_NIM_API_BASE`), and Azure (`AZURE_API_KEY`, `AZURE_API_BASE`,
`AZURE_API_VERSION`). If no known provider variable is set it logs a warning,
which you can ignore for Ollama.

### 3. Provider arguments (`model_args`)

Pass standard or provider-specific arguments through `model_args` (in config or
on the agent):

```python
model_args = {
    "temperature": 0.2,
    "max_tokens": 4096,
    "api_base": "http://localhost:11434",   # local deployments
}
```

The invoker always sends `drop_params=True`, so an argument a provider does not
support (say `presence_penalty` on Anthropic) is dropped instead of failing the
call. Request token log probabilities with `{"logprobs": True, "top_logprobs":
5}`; they land on `message.logprobs`. See LiteLLM's
[completion input](https://docs.litellm.ai/docs/completion/input) and
[responses input](https://docs.litellm.ai/docs/response_api/input) for every
argument.

### 4. Errors

LiteLLM maps provider errors to OpenAI-style exceptions (`RateLimitError`,
`ContextWindowExceededError`, `AuthenticationError`, `APIConnectionError`). The
invoker lets them propagate and the agent decides: a rate limit (`RateLimitError`
or HTTP 429) is retried with exponential backoff, `min(60, 2**n)` seconds plus up
to 1 s of jitter, up to `max_rate_limit_retries` times (default 8); other errors
are retried up to `max_llm_recall` times (default 0) before being re-raised. Parse failures never raise from
the invoker; they arrive as `execution_errors` and go to the prompt's exception
handler. Malformed JSON in tool-call arguments is reported the same way.

### 5. Costs

For every response the invoker records token usage plus dollar figures from
LiteLLM's pricing table (`response_cost`, `prompt_cost`, `completion_cost`, and
the per-token rates) on `message.usage`, which `message.cost`,
`session.cost`, and `dialog.cost` aggregate.

### 6. Chat Completions vs. Responses API

`api_type` selects the API (per agent, or `api_type:` in config):

- **`APITypes.COMPLETION`** (default) uses Chat Completions.
    - `Prompt.format` enables structured output: a Pydantic class is sent as
      `response_format`, a dict requests `{"type": "json_object"}`, and the reply
      is decoded with `json.loads` into `message.parsed`.
    - Function tools and MCP servers from the prompt are sent as `tools`.
- **`APITypes.RESPONSE`** uses OpenAI's Responses API.
    - `Prompt.format` is rejected (`ValueError`).
    - `addon_args={"web_search": True}` adds `{"type": "web_search_preview"}`,
      and `addon_args={"computer_use": {...}}` adds a `computer_use_preview`
      tool (display size and environment from the dict, defaulting to
      1280×800 `"browser"`).
    - `max_output_tokens` comes from `max_output_tokens` or
      `max_completion_tokens` in `model_args` (default 32000); `truncation` and
      `tool_choice` default to `"auto"`.
    - Reasoning summaries, when present, are stored in
      `message.metadata["reasoning"]`.

Both go through the same agent loop; only the wire format differs.

---

## Custom invokers

Any class that honours the contract can drive agents: a different SDK, a
local model server, a cache, or a scripted model for tests.

```python
from lllm import Message, Role
from lllm.runtimes.native import InvokeResult
from lllm.runtimes.native.invokers.base import BaseInvoker


class EchoInvoker(BaseInvoker):
    """Offline invoker: answers by echoing the last message."""

    def call(self, dialog, model, model_args=None, parser_args=None,
             responder="assistant", metadata=None, api_type=None, stream_handler=None):
        prompt = dialog.top_prompt
        content = f"echo: {dialog.tail.content}"
        errors, parsed = [], {}
        if prompt.parser is not None:
            try:
                parsed = prompt.parse(content, **(parser_args or {}))
            except Exception as exc:          # hand parse errors to the agent loop
                errors.append(exc)
        message = Message(
            role=Role.ASSISTANT,
            name=responder,
            content=content,
            parsed=parsed,
            model=model,
            metadata=dict(metadata or {}),
        )
        return InvokeResult(message=message, model_args=dict(model_args or {}),
                            execution_errors=errors)
```

Use it directly:

```python
from lllm import Agent, Prompt

agent = Agent(
    name="tester",
    system_prompt=Prompt(path="test/system", prompt="You are a test double."),
    model="echo",
    llm_invoker=EchoInvoker({}),
)
agent.open("main")
agent.receive("ping")
print(agent.respond().content)   # echo: ping
```

Or register a builder and select it from tactic config:

```python
from lllm import register_invoker

register_invoker("echo", lambda invoker_config: EchoInvoker(invoker_config))
```

```yaml
invoker: echo
invoker_config:        # optional; passed to the builder (default: the whole config)
  latency_ms: 0
agent_configs:
  - name: tester
    model_name: echo
    system_prompt: You are a test double.
```

`register_invoker(name, builder, *, overwrite=False)` raises if the name is
taken unless `overwrite=True`; `build_invoker(config)` looks up
`config["invoker"]` (default `"litellm"`) and calls the builder with
`config.get("invoker_config", config)`. Scripted invokers like this are the
easiest way to unit-test tactics without network access.
