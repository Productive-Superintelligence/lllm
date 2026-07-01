# Protocol Adapter

`NativeTacticAdapter` exposes a native object as a protocol tactic. The
service layer sees `TacticInfo`; native prompts, dialogs, sessions, and
invokers stay behind the adapter.

## Expose A Native Tactic

```python
from lllm.runtimes.native import NativeTacticAdapter

public_tactic = NativeTacticAdapter(
    native_tactic,
    package_ref="psi://demo/native/tactics/brief",
)
```

The adapter preserves the protocol boundary: callers send typed input and
receive typed output, while native code can still use agent groups, dialogs,
handlers, and sessions internally.

## Runtime-Owned Surroundings

The adapter can pass runtime-owned kwargs and call metadata without promoting
them into protocol concepts.

```python
from lllm import CallContext
from lllm.runtimes.native import NativeTacticAdapter

tactic = NativeTacticAdapter(
    native_tactic,
    run_kwargs={"tone": "precise"},
)

result = tactic.run(
    {"topic": "native services"},
    context=CallContext(trace_id="trace-1", metadata={"caller": "demo"}),
)
```

If the native method accepts `context`, it receives the `CallContext`. If it
accepts `metadata`, the adapter adds safe LLLM metadata such as request id,
trace id, refs, endpoint, and tags.

## Tactics As Tools

Native can consume a protocol tactic as a prompt tool:

```python
from lllm.runtimes.native import Prompt, tactic_as_function

lookup = tactic_as_function(search_tactic, parameter_mode="kwargs")
prompt = Prompt(
    path="research/answer",
    prompt="Use tools when needed.",
    function_list=[lookup],
)
```

Native tactics can also expose ordinary callable tools for Pydantic AI and
other runtimes:

```python
tool_callable = tactic.as_tool(parameter_mode="task")
```

For methods on a native tactic class, use `@tactictool` when a method should be
discoverable as a native tool resource. Selector refs can use `#tool_name` to
refer to a method-level tool.
