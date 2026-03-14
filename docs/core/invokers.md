# Invokers

Invokers are the execution engines of the LLLM framework. They abstract the underlying LLM APIs, taking the state of a `Dialog` and executing the API call to generate the next response.

The default invoker is built on [LiteLLM](https://github.com/BerriAI/litellm), giving you access to 100+ LLMs (OpenAI, Anthropic, Gemini, Vertex, local models via Ollama/vLLM, and more) through a single unified interface.

---

## The `BaseInvoker` Interface

All invokers inherit from `BaseInvoker` and implement a single method: `call`.

```python
class BaseInvoker(ABC):
    @abstractmethod
    def call(
        self,
        dialog: Dialog,
        model: str,
        model_args: Optional[Dict[str, Any]] = None,
        parser_args: Optional[Dict[str, Any]] = None,
        responder: str = 'assistant',
        metadata: Optional[Dict[str, Any]] = None,
        api_type: APITypes = APITypes.COMPLETION,
        stream_handler: BaseStreamHandler = None,
    ) -> InvokeResult:
        pass
```

The invoker returns an `InvokeResult` — a single object that bundles the clean `Message` (ready for dialog) with per-invocation diagnostics. This separation means the `Message` that goes into the dialog is free of debug data, while the agent loop has everything it needs for retry logic and analysis.

### Arguments

| Argument | Description |
|---|---|
| `dialog` | Current `Dialog`. The invoker reads `dialog.top_prompt` for tools, parser, and output format. |
| `model` | Model identifier (e.g. `'gpt-4o'`, `'anthropic/claude-3-5-sonnet'`). |
| `model_args` | Provider-specific arguments (`temperature`, `max_tokens`, `api_base`, etc.). |
| `parser_args` | Arguments passed to the prompt's output parser. |
| `responder` | Name of the agent generating the response (for logging and multi-agent routing). |
| `metadata` | Tracking metadata (e.g. frontend replay data). Attached to the `Message` but not sent to the LLM. |
| `api_type` | `COMPLETION` (Chat Completions) or `RESPONSE` (OpenAI Responses API). |
| `stream_handler` | Optional `BaseStreamHandler` for real-time streaming. |

### Return: `InvokeResult`

```python
@dataclass
class InvokeResult:
    raw_response: Any = None              # raw API response object
    model_args: Dict[str, Any] = ...      # actual args sent to the API
    execution_errors: List[Exception] = [] # parse/validation errors
    message: Optional[Message] = None     # the clean conversational message
```

The agent loop uses `invoke_result.has_errors` to decide whether to retry, and `invoke_result.message` for the dialog:

```python
invoke_result = invoker.call(dialog, model, ...)
if invoke_result.has_errors:
    # retry with exception handler
    raise AgentException(invoke_result.error_message)
dialog.append(invoke_result.message)
```

---

## Streaming with `BaseStreamHandler`

LLLM uses a callback pattern for streaming to keep dialog state clean and synchronous. Subclass `BaseStreamHandler`:

```python
class MyConsoleStreamer(BaseStreamHandler):
    def handle_chunk(self, chunk_content: str, chunk_response: Any):
        print(chunk_content, end="", flush=True)

# Pass it to the agent or invoker
invoke_result = invoker.call(dialog, model="gpt-4o", stream_handler=MyConsoleStreamer())
```

The invoker streams chunks to the handler in real-time while still returning the fully constructed `InvokeResult` at the end.

---

## Using the `LiteLLMInvoker`

The `LiteLLMInvoker` is the default invoker. It standardizes input and output schemas across all providers.

```python
from lllm.invokers.litellm import LiteLLMInvoker

invoker = LiteLLMInvoker()
```

### 1. Model Naming Convention

LiteLLM routes requests based on the model string prefix:

- **OpenAI:** `openai/gpt-4o` (or simply `gpt-4o`)
- **Anthropic:** `anthropic/claude-3-5-sonnet-20241022`
- **Google:** `gemini/gemini-1.5-pro`
- **Ollama (Local):** `ollama/llama3`
- **Azure:** `azure/my-gpt4-deployment`

See the [LiteLLM model list](https://models.litellm.ai/) and [LiteLLM providers](https://docs.litellm.ai/docs/providers) for all supported models.

### 2. Setting API Keys

LiteLLM automatically detects standard environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."
```

See [LiteLLM Python SDK](https://docs.litellm.ai/docs/#litellm-python-sdk) for environment variables per provider. Some providers (like Azure) require manual authentication setup — see [LiteLLM Proxy Authentication](https://docs.litellm.ai/docs/proxy_auth).

### 3. Provider-Specific Arguments (`model_args`)

Pass provider-specific or standard arguments via `model_args`. LLLM automatically drops unsupported parameters safely — if you pass `presence_penalty` to Anthropic (which doesn't support it), it's quietly dropped rather than throwing an error.

```python
model_args = {
    "temperature": 0.2,
    "max_tokens": 4096,
    "api_base": "http://localhost:11434",  # for local deployments
}
```

See [LiteLLM Completion API](https://docs.litellm.ai/docs/completion/input) and [LiteLLM Response API](https://docs.litellm.ai/docs/response_api/input) for all available arguments.

### 4. Exception Handling

LiteLLM standardizes all provider errors into the OpenAI exception format. Common exceptions:

- `RateLimitError` (HTTP 429)
- `ContextWindowExceededError` (HTTP 400)
- `AuthenticationError` (HTTP 401)
- `APIConnectionError` (HTTP 500)

The `Agent` class handles `max_llm_recall` retries and rate-limit backoffs automatically. Parse/validation errors are captured in `InvokeResult.execution_errors` and handled by the agent loop's exception handler.

### 5. Chat Completions vs. Responses API

The invoker supports two API modes via `api_type`:

- **`APITypes.COMPLETION`** (default): Uses Chat Completions. If `Prompt.format` is set (Pydantic model or JSON schema), structured output is enabled automatically.
- **`APITypes.RESPONSE`**: Uses OpenAI's Responses API. When the prompt enables `allow_web_search` or `computer_use_config`, these materialize as native OpenAI tools. Tool outputs are surfaced back through the interrupt handler.

Both modes go through the same agent call loop — the difference is only in how the invoker talks to the API.