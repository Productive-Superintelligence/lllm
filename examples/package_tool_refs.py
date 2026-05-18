"""
Package-style tool references without an LLM call.

Demonstrates:
  - Passing a coupled @tool Function directly to Prompt.function_list
  - Resolving a packaged @tool URL from Prompt.function_list
  - Auto-binding a decoupled prompt Function declaration to its implementation
  - Registering a tactic tool under pkg.tactics
  - Registering a proxy under pkg.proxies
  - Referencing those resources from Prompt.function_list and agent config tools

Run:
    python examples/package_tool_refs.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pydantic import BaseModel

from lllm import Function, Prompt, Tactic, tactictool, tool
from lllm.core.config import parse_agent_configs
from lllm.core.const import FunctionCall
from lllm.core.runtime import Runtime
from lllm.invokers import register_invoker
from lllm.proxies import BaseProxy, ProxyManager


register_invoker("noop", lambda config: object(), overwrite=True)


@tool(name="square", description="Square an integer.")
def square(n: int) -> int:
    return n * n


@tool(name="cube", description="Cube an integer.")
def cube(n: int) -> int:
    return n * n * n


@tool(name="negate", description="Negate an integer.")
def negate(n: int) -> int:
    return -n


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    value: str


class EchoTactic(Tactic):
    name = "echo"
    agent_group = []

    @tactictool("echo", description="Uppercase text.", config={"invoker": "noop"})
    def call(self, task: EchoInput) -> EchoOutput:
        return EchoOutput(value=task.text.upper())


class MathProxy(BaseProxy):
    _proxy_path = "math"
    _proxy_name = "Math Proxy"
    _proxy_description = "Small arithmetic API."

    @BaseProxy.endpoint(
        category="math",
        endpoint="double",
        description="Double an integer.",
        params={"n*": (int, 4)},
        response={"value": 8},
    )
    def double(self, params: dict) -> dict:
        return {"value": params["n"] * 2}


runtime = Runtime()
runtime.register_tool("square", square, namespace="shared.tools")
runtime.register_tool("cube", cube, namespace="shared.tools")
runtime.register_tactic("echo", EchoTactic, namespace="shared.tactics")
runtime.register_proxy("math", MathProxy, namespace="shared.proxies")


prompt = Prompt(
    path="demo/system",
    prompt="Use tools.",
    function_list=[
        negate,  # coupled style: schema and implementation travel together
        "shared.tools:square",  # coupled package style: implementation by URL
        Function(
            name="cube",  # decoupled style: declaration binds to shared.tools:cube
            description="Cube an integer.",
            properties={"n": {"type": "integer"}},
            required=["n"],
        ),
        "shared.tactics:echo",
    ],
)
prompt._qualified_key = "shared.prompts:demo/system"
resolved = prompt.resolve_function_refs(runtime)

negate_call = FunctionCall(id="call_0", name="negate", arguments={"n": 4})
square_call = FunctionCall(id="call_1", name="square", arguments={"n": 7})
cube_call = FunctionCall(id="call_2", name="cube", arguments={"n": 3})
echo_call = FunctionCall(id="call_3", name="echo", arguments={"text": "hello"})

print("Prompt tools:", sorted(resolved.functions))
print("negate(4):", resolved.functions["negate"](negate_call).result)
print("square(7):", resolved.functions["square"](square_call).result)
print("cube(3):", resolved.functions["cube"](cube_call).result)
print("echo('hello'):", resolved.functions["echo"](echo_call).result)


runtime.register_prompt(
    Prompt(path="system/agent", prompt="Use your configured tools."),
    namespace="app.prompts",
)
config = {
    "global": {
        "model_name": "noop-model",
        "tools": [
            "shared.tools:square",
            "shared.tools:cube",
            "shared.tactics:echo",
            "shared.proxies:math",
        ],
    },
    "agent_configs": [
        {
            "name": "assistant",
            "system_prompt_path": "app.prompts:system/agent",
        }
    ],
}

spec = parse_agent_configs(config, ["assistant"], "demo")["assistant"]
agent = spec.build(runtime, object())
agent_prompt = agent.system_prompt.resolve_function_refs(runtime)

print("Agent prompt functions:", sorted(agent_prompt.functions))

proxy_manager = ProxyManager(activate_proxies=["shared.proxies:math"], runtime=runtime)
print("CALL_API math.double:", proxy_manager("math.double", {"n": 5}))
