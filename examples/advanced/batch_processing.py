"""
Batch processing with bcall() and concurrent execution with ccall().

Demonstrates:
  - tactic.bcall()  — process a list of tasks in a thread pool (blocking)
  - tactic.ccall()  — async generator that yields results as they complete
  - Controlling max_workers for parallelism
  - Handling per-task failures gracefully with fail_fast=False

Usage:
    python examples/advanced/batch_processing.py

Prerequisites:
    export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY, etc.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from _api_key import MODEL  # noqa: E402

from lllm import Tactic
from lllm.logging import noop_store


# ---------------------------------------------------------------------------
# Simple one-agent tactic
# ---------------------------------------------------------------------------

class Classifier(Tactic):
    """Classify a piece of text into a category."""

    name = "classifier"
    agent_group = ["classifier"]

    def call(self, task: str) -> str:
        agent = self.agents["classifier"]
        agent.open("session")
        agent.receive(task)
        return agent.respond().content


config = {
    "global": {"model_name": MODEL},
    "agent_configs": [
        {
            "name": "classifier",
            "system_prompt": (
                "You are a text classifier. "
                "Classify the user's text into exactly ONE of these categories: "
                "Technology, Science, Sports, Politics, Entertainment. "
                "Reply with only the category name."
            ),
            "model_args": {"temperature": 0},
        }
    ],
}

tactic = Classifier(config, log_store=noop_store())

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

tasks = [
    "NASA's Artemis mission aims to return humans to the Moon by 2026.",
    "The championship final ended 3-2 after a dramatic penalty shootout.",
    "The new GPU architecture doubles performance per watt.",
    "Parliament passed the revised budget bill with a narrow majority.",
    "The sequel broke opening-weekend box office records worldwide.",
    "Researchers discovered a new exoplanet in the habitable zone.",
    "The central bank raised interest rates by 25 basis points.",
    "The open-source framework reached 100,000 GitHub stars.",
]

# ---------------------------------------------------------------------------
# 1. bcall — blocking batch, results in submission order
# ---------------------------------------------------------------------------

print("=== bcall (blocking, ordered results) ===\n")
results = tactic.bcall(tasks, max_workers=4)

for text, category in zip(tasks, results):
    print(f"  [{category:>15s}]  {text[:60]}...")

# ---------------------------------------------------------------------------
# 2. bcall with fail_fast=False — collect errors instead of raising
# ---------------------------------------------------------------------------

print("\n=== bcall (fail_fast=False) ===\n")
results_safe = tactic.bcall(tasks[:4], max_workers=2, fail_fast=False)

for text, result in zip(tasks[:4], results_safe):
    if isinstance(result, Exception):
        print(f"  ERROR: {result}")
    else:
        print(f"  [{result:>15s}]  {text[:60]}...")

# ---------------------------------------------------------------------------
# 3. ccall — async, yields results as they complete (out-of-order)
# ---------------------------------------------------------------------------

print("\n=== ccall (async, arrival order) ===\n")


async def run_concurrent():
    async for idx, result in tactic.ccall(tasks, max_workers=4):
        print(f"  task[{idx}] finished → [{result:>15s}]  {tasks[idx][:50]}...")


asyncio.run(run_concurrent())
