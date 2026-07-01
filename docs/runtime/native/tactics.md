# Native Tactics

The restored native `Tactic` is the callable runtime object used by native
workflows. It supports input/output validation, sync and async calls, batch
calls, sub-tactic composition, observer hooks, and a `TacticCallSession` for
each invocation.

## Agent-Backed Tactics

Subclass `NativeTactic` when a tactic should build live native agents from
config.

```python
from lllm.runtimes.native import NativeTactic


class BriefTactic(NativeTactic):
    name = "brief"
    input_type = str
    output_type = str
    agent_group = ["writer"]

    def call(self, task: str) -> str:
        writer = self.agents["writer"]
        writer.open("draft", prompt_args={"topic": task})
        writer.receive(task)
        return writer.respond().content
```

## Agent Config

```python
config = {
    "invoker": "litellm",
    "global": {
        "model_args": {"temperature": 0.1},
        "max_exception_retry": 3,
        "max_interrupt_steps": 5,
        "max_llm_recall": 0,
    },
    "agent_configs": [
        {
            "name": "writer",
            "model_name": "gpt-4o-mini",
            "system_prompt": "You write concise briefs about {topic}.",
        }
    ],
}

tactic = BriefTactic(config=config)
```

The parser reads these agent keys:

| Key | Meaning |
| --- | --- |
| `name` | Agent name. Must match an entry in `agent_group`. |
| `model_name` | Provider model id passed to the invoker. |
| `system_prompt` | Inline prompt string. |
| `system_prompt_path` | Prompt resource path loaded from the native runtime. |
| `api_type` | `completion` or `response`. |
| `model_args` | Provider args such as temperature, token limits, headers, or skills payloads. |
| `max_exception_retry` | Parser/validation repair cap. |
| `max_interrupt_steps` | Tool-call loop cap. |
| `max_llm_recall` | Provider/API error recall cap. |
| `tools` | Tool, tactic, or proxy refs to add to the prompt. |
| `proxy` | Proxy manager and execution-tool settings. |
| `context_manager` | Built-in or registered context manager config. |
| `skills` | Local, URL, wildcard, or provider-hosted skill entries. |
| `extra_settings` | Native-specific settings reserved for tactic code. |

## TacticCallSession

`NativeTactic` creates a fresh tracked agent group per call. The internal
`_TrackedAgent` wrapper records each `agent.respond()` call into the current
`TacticCallSession`, so `session.summary()` includes agent-call counts and
total cost.

```python
session = tactic("Summarize the native runtime.", return_session=True)

print(session.state)
print(session.summary())
print(session.agent_sessions["writer"][0].delivery.content)
```

## Batch And Async Calls

Native tactics preserve compatibility helpers such as `run`, `arun`, `bcall`,
and `ccall`. Optional `stream`, `astream`, `events`, and `aevents` surfaces are
reported only when a tactic implementation actually provides them.
