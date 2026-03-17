# Agent

An `Agent` is LLLM's "caller" — it holds a system prompt and a base model, and executes prompts through a deterministic call loop. It is **not** a long-running process; it is stateless per call. The tactic creates fresh agent instances for each invocation and passes them the dialogs to work on.

```python
agent.open("task_1", prompt_args={"topic": topic})  # create a dialog, seed with system prompt
agent.receive("Analyze this paper")                  # append user turn
response = agent.respond()                           # run the call loop, return Message
```

---

## Declaration by Configuration

Agents are declared in YAML, not in Python. A config entry specifies everything about an agent: which model it uses, which system prompt it reads, and how its call loop behaves.

```yaml
# configs/research_writer.yaml
agent_group_configs:
  researcher:
    model_name: gpt-4o
    system_prompt_path: researcher_system   # loaded from the prompt registry
    temperature: 0.3
    max_completion_tokens: 4000
    max_exception_retry: 3

  writer:
    model_name: gpt-4o
    system_prompt_path: writer_system
    temperature: 0.7
```

`build_tactic` reads this config, resolves each `system_prompt_path` from the runtime, and constructs live `Agent` instances at call time:

```python
from lllm import build_tactic, resolve_config

config = resolve_config("research_writer")
tactic = build_tactic(config, ckpt_dir="./runs")

# inside tactic.call():
researcher = self.agents["researcher"]   # Agent instance, ready to use
```

For the full config schema — global defaults, `model_args`, inheritance via `base`, and multi-package vendoring — see [Configuration](config.md).

---

## LLM Call vs. Agent Call

| | LLM Call | Agent Call |
| --- | --- | --- |
| Input | Flat list of chat messages. | `Dialog` owned by the agent, seeded with a system `Prompt`. |
| Output | Raw model string plus metadata. | `AgentCallSession` — delivered message, invoke traces, full diagnostics. |
| Responsibility | Caller decides whether to retry, parse, or continue. | Agent handles retries, parsing, exception recovery, and interrupts. |
| Determinism | Best-effort. | Guaranteed next state or explicit exception. |

A core philosophy of LLLM is to treat the agent as a "function" — the goal of the agent call is to make it as stable and deterministic as possible.

![Agent call state machine](../assets/agent_call.png)

---

## The Call Loop

`agent.respond()` runs a state machine that advances until it reaches a terminal state:

1. The invoker calls the LLM and returns an `InvokeResult`.
2. If parsing failed (`invoke_result.has_errors`), the error is recorded and the prompt's **exception handler** generates a retry message. The dialog is forked for the retry so recovery messages never pollute the canonical history.
3. If the LLM returned tool calls, each function is executed, results are fed back via the **interrupt handler**, and the loop continues.
4. If the LLM returned a plain assistant message, the loop transitions to `"success"` and the message becomes `session.delivery`.
5. Network/API errors trigger backoff and LLM recall retries (up to `max_llm_recall`).

### Key types

**`InvokeResult`** — returned by the invoker per LLM call:

```python
@dataclass
class InvokeResult:
    raw_response: Any                   # raw API response
    model_args: Dict[str, Any]          # actual args sent to the API
    execution_errors: List[Exception]   # parse/validation errors
    message: Optional[Message]          # the clean conversational message
```

**`AgentCallSession`** — tracks the full lifecycle:

```python
class AgentCallSession(BaseModel):
    agent_name: str
    state: Literal["initial", "exception", "interrupt", "llm_recall", "success", "failure"]
    exception_retries: Dict[str, List[Exception]]
    interrupts: Dict[str, List[FunctionCall]]
    llm_recalls: Dict[str, List[Exception]]
    invoke_traces: Dict[str, List[InvokeResult]]  # all invocations per step
    delivery: Optional[Message]                   # final message on success
```

---

## Interrupt and Exception Handling

Each `Prompt` can specify inline handlers that the call loop uses automatically:

- **`on_exception`** — receives `{error_message}` whenever parsing or validation fails. Called up to `max_exception_retry` times. The dialog is forked for each retry.
- **`on_interrupt`** — receives `{call_results}` after function execution. Remains in the dialog for transparency.
- **`on_interrupt_final`** — fires when the agent hits `max_interrupt_steps`, prompting a natural-language summary.

Handlers inherit the prompt's parser, tools, and allowed functions, so a single prompt definition covers the entire agent loop.

---

## Dialog Management

Agents own the dialogs they create, keyed by a user-chosen alias. The alias makes code self-documenting and prevents dialogs from leaking between agents.

```python
agent = self.agents["coder"]

# Open a dialog — seeds with system prompt, becomes active
agent.open("task_1", prompt_args={"language": "Python"})
agent.receive("Write a sorting function")
response = agent.respond()

# Open a second dialog, switch between them
agent.open("task_2", prompt_args={"language": "Rust"})
agent.switch("task_1")
agent.receive("Now add error handling")
response = agent.respond()

# Fork a dialog for exploration (last_n messages carry over)
agent.fork("task_1", "task_1_alt", last_n=2, switch=True)
agent.receive("Try a different algorithm")
response = agent.respond()
```

For multi-agent collaboration, each agent maintains its own dialog. Information is shared explicitly by passing content between agents in the tactic:

```python
coder.open("collab", prompt_args={...})
reviewer.open("collab", prompt_args={...})

coder.receive("Write a REST API")
code = coder.respond()

reviewer.receive(code.content, name=coder.name)
review = reviewer.respond()

coder.receive(review.content, name=reviewer.name)
revision = coder.respond()
```

---

## Diagnostics

Pass `return_session=True` to get the full `AgentCallSession` alongside the message:

```python
class ResearchTactic(Tactic):
    def call(self, task: str, **kwargs):
        agent = self.agents["researcher"]
        agent.open("research", prompt_args={"topic": task})

        session = agent.respond(return_session=True)

        print(session.delivery.parsed)          # structured output (if Pydantic format)
        print(session.delivery.cost)            # token costs
        print(len(session.invoke_traces))       # number of interrupt steps taken
        print(agent.current_dialog.tree_overview())  # dialog tree structure

        return session.delivery.content
```
