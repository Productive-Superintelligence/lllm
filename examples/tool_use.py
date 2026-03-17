"""
Tool calling with the @tool decorator.

Demonstrates:
  - Defining tools with @tool (auto-infers schema from type hints)
  - Attaching tools to a Prompt via function_list
  - Agent automatically calling tools and feeding results back

Prerequisites:
    export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY, etc.
"""
import math

from lllm import Tactic, Prompt
from lllm.core.prompt import tool


# ---------------------------------------------------------------------------
# Define tools — @tool turns a plain function into a Function object
# ---------------------------------------------------------------------------

@tool(
    description="Compute the square root of a non-negative number.",
    prop_desc={"x": "The number to take the square root of"},
)
def sqrt(x: float) -> str:
    if x < 0:
        return "Error: square root of a negative number is not real."
    return f"{math.sqrt(x):.6f}"


@tool(
    description="Compute the factorial of a non-negative integer n.",
    prop_desc={"n": "A non-negative integer"},
)
def factorial(n: int) -> str:
    if n < 0:
        return "Error: factorial is only defined for non-negative integers."
    return str(math.factorial(n))


@tool(
    description="Convert degrees to radians.",
    prop_desc={"degrees": "Angle in degrees"},
)
def deg_to_rad(degrees: float) -> str:
    return f"{math.radians(degrees):.6f}"


# ---------------------------------------------------------------------------
# Build a prompt that exposes the tools to the agent
# ---------------------------------------------------------------------------

math_prompt = Prompt(
    path="math/assistant",
    prompt=(
        "You are a math assistant. "
        "Use the provided tools to answer questions precisely. "
        "Always show the tool result in your final answer."
    ),
    function_list=[sqrt, factorial, deg_to_rad],
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

agent = Tactic.quick(system_prompt=math_prompt)
agent.open("session")

agent.receive(
    "What is the square root of 1764? "
    "Also compute 8 factorial. "
    "And what is 45 degrees in radians?"
)

# return_session=True gives back the full AgentCallSession with diagnostics
agent_session = agent.respond(return_session=True)
response = agent_session.delivery

print(response.content)

# Diagnostics: how many tool-call rounds happened
print(f"\nTool-call rounds: {len(agent_session.interrupts)}")
print(f"Total cost:       {agent_session.cost}")
