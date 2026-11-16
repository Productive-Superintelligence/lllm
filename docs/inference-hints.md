# Inference hints

An agent knows things a model server cannot see in a single request: which
calls belong to one line of reasoning, which dialogs were forked from
which and so share a prompt prefix, how urgent a call is, and roughly how
long the answer will be. `InferenceHints` sends that with every model call,
in the form NVIDIA Dynamo, vLLM, or SGLang reads, so a KV-cache-aware
router can keep a dialog on the worker that holds its cache, warm the cache
for the next turn, and serve urgent work first.

The session identity comes from the dialogs themselves, so there is
nothing to track by hand:

```python title="hints.py"
from lllm import Agent, InferenceHints, Prompt
from lllm.runtimes.native.invokers import build_invoker

agent = Agent(
    name="planner",
    system_prompt=Prompt(path="planner/system", prompt="You plan studies."),
    model="openai/Qwen/Qwen2.5-7B-Instruct",
    llm_invoker=build_invoker({"invoker": "litellm"}),
    model_args={
        "api_base": "http://localhost:8000/v1",
        "api_key": "local",
        "max_tokens": 256,
    },
    inference_hints=InferenceHints(profile="dynamo", priority=5),  # (1)!
)


def hint_fields(session):
    """What the call sent, with each dialog id replaced by its alias."""
    sent = session.invoke_traces[0][0].model_args  # (2)!
    alias = {d.dialog_id: name for name, d in agent.dialogs.items()}
    headers = {k: alias[v] for k, v in sent["extra_headers"].items()}
    return headers, sent["extra_body"]


agent.open("main")
agent.receive("Plan a study of prefix caching.")
headers, body = hint_fields(agent.respond(return_session=True))
print(headers)
#> {'x-dynamo-session-id': 'main'}
print(body)  # (3)!
#> {'nvext': {'agent_hints': {'priority': 5, 'osl': 256}}}

agent.receive("List the risks of the prefix caching study.")
headers, _ = hint_fields(agent.respond(return_session=True))
print(headers)  # (4)!
#> {'x-dynamo-session-id': 'main'}

agent.fork("main", "small")
agent.receive("Adapt the plan for a 1B model.")
headers, _ = hint_fields(agent.respond(return_session=True))
print(headers)  # (5)!
#> {'x-dynamo-session-id': 'small', 'x-dynamo-parent-session-id': 'main'}
```

1. Hints are off unless you set them. `profile` picks the serving stack.
2. `invoke_traces` records the arguments each model call sent to LiteLLM,
   and the hints are in `extra_headers` and `extra_body`. Dialog ids are
   random hex strings, so the helper prints aliases in their place.
3. `osl`, the expected output length, defaults to the agent's
   `max_tokens`.
4. The second turn of a dialog is the same session, so the router can send
   it to the worker that already holds the first turn's KV cache.
5. A forked dialog is a new session that names its parent, which tells the
   router it starts with the parent's prefix.

## Turning hints on

Pass `inference_hints=` to an `Agent`, or set `inference_hints` in a
config, under `global` or per agent. The two are merged like any other
dict, so a tactic can share a profile and give each agent its own
priority:

```yaml
global:
  model_name: openai/Qwen/Qwen2.5-7B-Instruct
  model_args: {api_base: "http://localhost:8000/v1", max_tokens: 512}
  inference_hints: {profile: dynamo, session_prefix: run-42}
agent_configs:
  - name: planner        # the user is waiting on it
    system_prompt: You plan.
    inference_hints: {priority: 10, speculative_prefill: true}
  - name: summarizer     # background work
    system_prompt: You summarize.
    inference_hints: {priority: 1}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `profile` | `openai` | The serving stack: `dynamo`, `vllm`, `sglang`, or `openai` |
| `session` | `True` | Send the dialog's session identity |
| `session_prefix` | | Put a prefix on session ids, such as a run id: `run-42:<dialog id>` |
| `priority` | | How urgent the call is; higher is more urgent |
| `strict_priority` | | Dynamo's router queue tier; higher tiers always go first |
| `osl` | | Expected output length in tokens (Dynamo); defaults to `max_tokens` or `max_completion_tokens` |
| `speculative_prefill` | | Ask Dynamo to warm the cache for the next turn |
| `cache_salt` | | vLLM prefix-cache salt, to keep tenants' cached prefixes apart |

Fields you leave unset are not sent, and an unknown field is a validation
error. Hints are computed once per `respond()` and sent with every model
call it makes, including tool rounds and repairs.

## Sessions follow dialogs

A dialog is one line of reasoning, so it is one session: every turn of a
dialog carries the same id, `open()` starts a new session, and
`fork(alias, child_alias)` starts a child session that names its parent.
A router that keeps sessions on one worker reuses the dialog's KV cache
turn after turn, and a fork can be placed next to its parent's prefix.

Session ids are dialog ids, with `session_prefix` in front when you set
one. `session=False` sends no identity, only the other fields. Inside a
tactic, every call builds fresh agents, so each call's dialogs are new
sessions.

## What each server receives

Each profile writes the same knowledge in its server's own fields:

| Profile | Session headers | Body fields |
| --- | --- | --- |
| `dynamo` | `x-dynamo-session-id`, `x-dynamo-parent-session-id` | `nvext.agent_hints`: `priority`, `strict_priority`, `osl`, `speculative_prefill` |
| `vllm` | `x-session-id`, `x-parent-session-id` | `priority`, `cache_salt` |
| `sglang` | `x-session-id`, `x-parent-session-id` | `priority` |
| `openai` | `x-session-id`, `x-parent-session-id` | none |

`request_fields()` returns what one call would add, which is handy for
checking a profile:

```python title="profiles.py"
from lllm import InferenceHints

vllm = InferenceHints(profile="vllm", priority=2, cache_salt="tenant-a")
fields = vllm.request_fields(dialog_id="d2", parent_dialog_id="d1")
print(fields["extra_headers"])
#> {'x-session-id': 'd2', 'x-parent-session-id': 'd1'}
print(fields["extra_body"])  # (1)!
#> {'priority': -2, 'cache_salt': 'tenant-a'}

sglang = InferenceHints(profile="sglang", priority=2)
print(sglang.request_fields(dialog_id="d2")["extra_body"])
#> {'priority': 2}
```

1. In LLLM a higher priority is more urgent, as in Dynamo and SGLang. vLLM
   runs lower values first, so the `vllm` profile flips the sign.

The `openai` profile adds only headers, which servers that do not use them
ignore, so it is safe with any endpoint. Body fields are not: vLLM rejects
a nonzero `priority` unless it runs with priority scheduling. Explicit
`extra_headers` or `extra_body` in `model_args` always win over hints.

## On the server side

Hints change behavior only where the serving stack is set up to use them:

| Stack | What to enable |
| --- | --- |
| NVIDIA Dynamo | KV-aware routing; `--router-session-affinity-ttl-secs` to keep sessions on one worker; `--router-queue-threshold` for priority queueing; `--router-track-output-blocks` to use `osl` |
| vLLM | `--scheduling-policy priority` for `priority`; prefix caching reuses shared prefixes |
| SGLang | `--enable-priority-scheduling` for `priority`; `--radix-eviction-policy priority` to evict low-priority cache first |

Dynamo also reads the generic `x-session-id` headers the other profiles
send, and it passes `priority` on to its backend engines in each engine's
own convention.

## Ending a session

Dynamo can release a session's state as soon as the session ends, instead
of waiting for a timeout, when a final request carries
`x-dynamo-session-final`. `final_request_fields()` builds its fields:

```python title="final.py"
from lllm import InferenceHints

hints = InferenceHints(profile="dynamo")
print(hints.final_request_fields("d1")["extra_headers"])
#> {'x-dynamo-session-id': 'd1', 'x-dynamo-session-final': 'true'}
```

The agent does not send it for you. When a dialog is done, send a minimal
request with these fields, for example with `litellm.completion(...,
max_tokens=1, **fields)`, using the dialog's `dialog_id`.

## Reference

`InferenceHints(profile="openai", session=True, session_prefix=None,
priority=None, strict_priority=None, osl=None, speculative_prefill=None,
cache_salt=None)` is a Pydantic model, importable from `lllm`.

| Method | Returns |
| --- | --- |
| `request_fields(*, dialog_id=None, parent_dialog_id=None, model_args=None)` | `{"extra_headers": ..., "extra_body": ...}` for one call; `model_args` supplies the `osl` default |
| `final_request_fields(dialog_id)` | The same, plus Dynamo's end-of-session header |
| `session_ids(dialog_id, parent_dialog_id=None)` | The session and parent ids that would be sent |

See [Models](models.md#local-and-openai-compatible-servers) for pointing
agents at your own server, and [Dialogs](dialogs.md) for forking.
