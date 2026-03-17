"""
Multi-turn conversation with dialog management.

Demonstrates:
  - agent.open() / receive() / respond() pattern
  - Maintaining conversation history across turns
  - Forking a dialog to explore a different branch

Prerequisites:
    export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY, etc.
"""
from lllm import Tactic


# Create an agent that remembers previous messages
agent = Tactic.quick(
    system_prompt="You are a concise Python tutor. Keep answers to 2-3 sentences.",
    model="gpt-4o",
)

agent.open("lesson")

# Turn 1
agent.receive("What is a list comprehension in Python?")
r1 = agent.respond()
print("Turn 1:", r1.content)

# Turn 2 — agent remembers previous turns
agent.receive("Can you show me a simple example?")
r2 = agent.respond()
print("\nTurn 2:", r2.content)

# Turn 3
agent.receive("How does it compare to a regular for loop?")
r3 = agent.respond()
print("\nTurn 3:", r3.content)

# Fork the dialog at this point and explore a different direction
# The fork shares history up to here but diverges from now on
agent.fork("lesson", "dict_branch")

agent.receive("Now explain dictionary comprehensions the same way.")
r4 = agent.respond()
print("\nForked branch — Turn 4:", r4.content)

# Switch back to the original branch to show they are independent
agent.switch("lesson")
agent.receive("What about generator expressions?")
r5 = agent.respond()
print("\nOriginal branch — Turn 5:", r5.content)

# Print token usage for the last response
print(f"\nLast response usage: {r5.usage}")
