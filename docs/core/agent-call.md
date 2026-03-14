# Agent Call

Agent calls are the defining concept of LLLM. Instead of exposing raw LLM completions, every agent implements a deterministic state machine that **must** transition from an initial dialog to a well-defined output state (or raise an error). This contract keeps downstream systems simple—no consumer needs to guess whether the model "felt done".

A core philosophy of LLLM is to treat the agent as a "function", and the goal of the agent call is to make it as stable and deterministic as possible.

![Agent call state machine](../assets/agent_call.png)

## LLM Call vs. Agent Call

| | LLM Call | Agent Call |
| --- | --- | --- |
| Input | Flat list of chat messages. | `Dialog` owned by the agent, seeded with a system `Prompt`. |
| Output | Raw model string plus metadata. | `AgentCallSession` containing the delivered message, invoke traces, and full diagnostics. |
| Responsibility | Caller decides whether to retry, parse, or continue. | Agent handles retries, parsing, exception recovery, and interrupts until it reaches the desired state. |
| Determinism | Best-effort. | Guaranteed next state or explicit exception. |

The `Agent` dataclass in `lllm/core/agent.py` manages dialogs via aliases and exposes `respond(alias)` for the common case and `_call(dialog)` for the full loop. `Orchestrator` wraps these calls with logging and a user-friendly `.call(task)` signature. Under the hood each agent delegates to an invoker implementation (`lllm.invokers.BaseInvoker`) so the same loop can target OpenAI's Chat Completions API or the Responses API by toggling the `api_type` field.

## Key Types

### `InvokeResult`

Returned by the invoker for each LLM call. Bundles the clean `Message` (ready for dialog) with per-invocation diagnostics:

```python
@dataclass
class InvokeResult:
    raw_response: Any = None          # raw API response object
    model_args: Dict[str, Any] = ...  # actual args sent to the API
    execution_errors: List[Exception]  # parse/validation errors
    message: Optional[Message] = None  # the conversational message
```

The agent loop checks `invoke_result.has_errors` to decide whether to retry, and appends `invoke_result.message` to the dialog on success.

### `AgentCallSession`

Tracks the entire agent call lifecycle — every retry, every tool interrupt, every LLM recall:

```python
class AgentCallSession(BaseModel):
    agent_name: str
    state: Literal["initial", "exception", "interrupt", "llm_recall", "success", "failure"]
    
    # Per-interrupt-step tracking
    exception_retries: Dict[str, List[Exception]]
    interrupts: Dict[str, List[FunctionCall]]
    llm_recalls: Dict[str, List[Exception]]
    invoke_traces: Dict[str, List[InvokeResult]]  # all invoke results at each step
    
    delivery: Optional[Message] = None  # the final delivered message on success
```

On success, `session.delivery` holds the final message. `invoke_traces` records every invocation (including retries) for debugging and analysis.

## State Machine Lifecycle

1. **Open dialog** — `agent.open('task_1', prompt_args={...})` creates a dialog seeded with the agent's system prompt and registers it under the alias `'task_1'`.

2. **Send user message** — `agent.receive("Summarize this document")` appends a user-role message to the active dialog.

3. **Respond** — `agent.respond()` runs the agent call loop:
   - The invoker calls the LLM and returns an `InvokeResult`.
   - If parsing failed (`invoke_result.has_errors`), the error is recorded in `AgentCallSession` and the prompt's exception handler generates a retry message.
   - If the LLM returned tool calls, each function is executed, results are fed back via the interrupt handler, and the loop continues.
   - If the LLM returned a plain assistant message, the session transitions to `"success"` and the message is set as `session.delivery`.
   - Network/runtime errors trigger backoff and LLM recall retries.

4. **Completion** — `respond()` returns the delivered `Message`. If you need full diagnostics, pass `return_session=True` to get the `AgentCallSession`.

## Interrupt vs. Exception Handling

Each `Prompt` can specify inline handlers:

- **Exception handler** (`Prompt.on_exception(session)`) receives `{error_message}` whenever parsing or validation fails. The dialog is forked for each retry, so exception-recovery messages never leak into the canonical dialog.
- **Interrupt handler** (`Prompt.on_interrupt(session)`) receives `{call_results}` after function execution. These messages remain in the dialog for transparency.
- **Final interrupt handler** (`Prompt.on_interrupt_final(session)`) fires when the agent hits `max_interrupt_steps`, prompting a natural-language summary.

Handlers inherit the prompt's parser, tools, and allowed functions, so a single definition covers the entire agent loop.

## Function Calls, MCP, and Tools

`Prompt` instances can bundle:

- `Function` objects (structured JSON schemas) that wrap Python callables.
- MCP server descriptors for Model Context Protocol tools.
- Optional `allow_web_search` and `computer_use_config` hints for provider-native toolchains.

During an agent call, every function call is tracked as a `FunctionCall` object with `arguments`, `result`, `result_str`, and `error_message`. The loop prevents duplicate calls by checking `FunctionCall.is_repeated`.

### Selecting the LLM API

Each agent entry in `agent_configs` accepts `api_type`:

```toml
[agent_configs.researcher]
model_name = "gpt-4o-mini"
system_prompt_path = "research/system"
api_type = "response"  # or "completion"
```

- `completion` (default) uses Chat Completions. If `Prompt.format` is set, structured output is enabled automatically.
- `response` opts into OpenAI's Responses API, enabling native web search and computer-use tools when available.

## Agent Dialog Management

Agents own the dialogs they create. Each dialog is keyed by a user-chosen alias:

```python
agent = self.agents["coder"]

# Open a dialog — seeds with system prompt, becomes active
agent.open('task_1', prompt_args={'language': 'Python'})

# Send messages to the active dialog
agent.receive("Write a sorting function")
response = agent.respond()

# Open a second dialog, switch between them
agent.open('task_2', prompt_args={'language': 'Rust'})
agent.switch('task_1')  # go back to first dialog
agent.receive("Now add error handling")
response = agent.respond()

# Fork a dialog for exploration
agent.fork('task_1', 'task_1_alt', last_n=2, switch=True)
agent.receive("Try a different algorithm")
response = agent.respond()
```

The alias makes code self-documenting — `agent.switch('planning')` communicates intent more clearly than `agent.dialog = some_variable`.

### Why dialogs are owned by agents

Because every dialog starts with one agent's system prompt, it is semantically owned by that agent. Letting `agentB` accidentally call on `agentA`'s dialog (which carries `agentA`'s system prompt) is a subtle bug. Internalizing dialogs to the agent that created them makes ownership explicit.

For multi-agent conversations, each agent maintains its own dialog, and developers explicitly pass messages between them:

```python
coder.open('collab', prompt_args={...})
reviewer.open('collab', prompt_args={...})

coder.receive("Write a REST API")
code = coder.respond()

# Pass coder's output to reviewer
reviewer.receive(code.content, name=coder.name)
review = reviewer.respond()

# Pass review back to coder
coder.receive(review.content, name=reviewer.name)
revision = coder.respond()
```

## Implementing Custom Agents

To ship a new agent:

1. Subclass `Orchestrator`, set `agent_type` and `agent_group`.
2. Implement `call(self, task, **kwargs)` to orchestrate dialogs.

```python
class SimpleAgent(Orchestrator):
    agent_type = "simple"
    agent_group = ["assistant"]
    
    def call(self, task: str, **kwargs):
        agent = self.agents["assistant"]
        agent.open('main', prompt_args={'user_input': task})
        return agent.respond().content
```

Agents register themselves automatically through `Orchestrator.__init_subclass__`, so once your class is imported it becomes available to `build_agent`.

## Full Example with Diagnostics

```python
class ResearchAgent(Orchestrator):
    agent_type = "researcher"
    agent_group = ["researcher"]
    
    def call(self, task: str, **kwargs):
        agent = self.agents["researcher"]
        agent.open('research', prompt_args={'topic': task})
        
        # Get full session for diagnostics
        session = agent.respond(return_session=True)
        
        # Inspect results
        print(session.delivery.parsed)           # structured output
        print(session.delivery.cost)             # token costs
        print(len(session.invoke_traces))        # number of interrupt steps
        print(agent.current_dialog.tree_overview())  # dialog tree structure
        
        return session.delivery.content
```