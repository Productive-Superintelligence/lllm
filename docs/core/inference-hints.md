# Inference Hints

An agent knows things an inference server cannot see from a single request:
which calls belong to the same reasoning chain, which chains were forked from
which (and so share a prompt prefix), how urgent a call is, and roughly how
long the answer will be. Serving stacks with KV-cache-aware routing and
priority scheduling can use exactly that: keep a chain on the worker that
already holds its prefix, warm the cache for the next turn, and schedule
urgent work first.

`InferenceHints` lets a native agent send that knowledge with every model
call. Hints are off unless you configure them.

## Turn them on

On an agent:

```python
from lllm import Agent, InferenceHints, Prompt
from lllm.runtimes.native.invokers import build_invoker

agent = Agent(
    name="planner",
    system_prompt=Prompt(path="planner/system", prompt="You plan."),
    model="hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
    llm_invoker=build_invoker({"invoker": "litellm"}),
    model_args={"api_base": "http://127.0.0.1:8000/v1", "max_tokens": 512},
    inference_hints=InferenceHints(profile="dynamo", priority=5, speculative_prefill=True),
)
```

Or in an agent config, so every agent of a tactic gets them:

```yaml
global:
  model_name: hosted_vllm/Qwen/Qwen2.5-7B-Instruct
  model_args:
    api_base: http://127.0.0.1:8000/v1
  inference_hints:
    profile: dynamo
    priority: 5
    speculative_prefill: true

agent_configs:
  - name: planner
    system_prompt: You plan.
```

## Session identity comes from dialogs

A dialog is one reasoning chain, so it is one session: every turn of an
agent's dialog carries the same session id. A dialog created with
`agent.fork(...)` names its parent, which tells the server that the child
starts from the parent's prefix.

```python
agent.open("main")
agent.receive("Draft a plan.")
agent.respond()                  # session = main dialog
agent.fork("main", "retry")      # child dialog shares main's prefix
agent.respond()                  # session = retry, parent = main
```

Session ids are the dialogs' ids. Set `session_prefix` (for example a run id)
to get `run-42:<dialog id>`, or `session=False` to send no identity.

## Profiles

| Profile | Session identity | Body fields |
| --- | --- | --- |
| `dynamo` | `x-dynamo-session-id`, `x-dynamo-parent-session-id` | `nvext.agent_hints`: `priority`, `strict_priority`, `osl`, `speculative_prefill` |
| `vllm` | `x-session-id`, `x-parent-session-id` | `priority`, `cache_salt` |
| `sglang` | `x-session-id`, `x-parent-session-id` | `priority` |
| `openai` (default) | `x-session-id`, `x-parent-session-id` | none |

| Field | Meaning |
| --- | --- |
| `priority` | Higher is more urgent. The `vllm` profile converts it to vLLM's order, where lower runs first. |
| `strict_priority` | Dynamo router queue tier; higher tiers always go first. |
| `osl` | Expected output tokens (Dynamo). Defaults to the agent's `max_tokens` or `max_completion_tokens`. |
| `speculative_prefill` | Ask Dynamo to prefill the predicted next-turn prefix after this turn. |
| `cache_salt` | vLLM prefix-cache salt, to keep tenants' cached prefixes apart. |

Unset fields are not sent. The `openai` profile only adds headers, which
servers that do not use them ignore, so it is safe against any endpoint;
send `priority` to vLLM or SGLang only when priority scheduling is enabled
there, because they reject it otherwise. Explicit `extra_headers` or
`extra_body` in `model_args` always win over hints.

## Server side

Hints change behavior only where the serving stack is configured to use them:

| Stack | What to enable |
| --- | --- |
| NVIDIA Dynamo | KV-aware routing; `--router-queue-threshold` for priority queueing; `--router-track-output-blocks` to use `osl` |
| vLLM | `--scheduling-policy priority` for `priority`; prefix caching for shared prefixes |
| SGLang | `--enable-priority-scheduling`; `--radix-eviction-policy priority` for priority-aware cache eviction |

Session headers matter for routers with session affinity, which keep a chain
on one worker so its KV cache is reused. Dynamo also accepts the generic
`x-session-id` headers that the other profiles send.

## Ending a session

Dynamo can release per-session state as soon as a chain is finished. Build a
minimal request with `final_request_fields`:

```python
hints = InferenceHints(profile="dynamo")
fields = hints.final_request_fields(agent.dialogs["main"].tree_node.dialog_id)
# {"extra_headers": {"x-dynamo-session-id": "...", "x-dynamo-session-final": "true"}}
```

## What the server receives

Against a local OpenAI-compatible server, the two calls above arrive as:

```json
{"headers": {"x-dynamo-session-id": "run-1:714fa934..."},
 "nvext": {"agent_hints": {"priority": 5, "osl": 24, "speculative_prefill": true}}}
{"headers": {"x-dynamo-session-id": "run-1:be2272f7...",
             "x-dynamo-parent-session-id": "run-1:714fa934..."},
 "nvext": {"agent_hints": {"priority": 5, "osl": 24, "speculative_prefill": true}}}
```

and with `InferenceHints(profile="vllm", priority=2, cache_salt="tenant-a")`:

```json
{"headers": {"x-session-id": "08b7512d..."}, "priority": -2, "cache_salt": "tenant-a"}
```

The second Dynamo request carries the forked dialog's shared prefix (four
messages) and names its parent: the structure a KV-aware router reuses.

## Reference

`InferenceHints(profile="openai", session=True, session_prefix=None,
priority=None, strict_priority=None, osl=None, speculative_prefill=None,
cache_salt=None)`

- `request_fields(*, dialog_id=None, parent_dialog_id=None, model_args=None)`
  returns `{"extra_headers": ..., "extra_body": ...}` for one call.
- `final_request_fields(dialog_id)` adds Dynamo's end-of-session header.
- `session_ids(dialog_id, parent_dialog_id=None)` returns the ids that would
  be sent.

See [Agent](agent.md) for where hints sit in the call loop, and the
[quick start](../tutorials/01-quick-start.md) for pointing agents at a local
server.
