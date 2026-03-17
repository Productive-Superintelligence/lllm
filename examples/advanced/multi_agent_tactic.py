"""
Custom Tactic with multiple agents (two-stage pipeline).

Demonstrates:
  - Subclassing Tactic for multi-agent orchestration
  - Declaring agents with inline system prompts via config
  - Passing the output of one agent as input to the next
  - Suppressing the LogStore warning with noop_store()

Usage:
    python examples/advanced/multi_agent_tactic.py

Prerequisites:
    export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY, etc.
"""
import sys
import os

# Allow running from the repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from _api_key import MODEL  # noqa: E402  (relative to this folder)

from lllm import Tactic
from lllm.logging import noop_store


# ---------------------------------------------------------------------------
# Custom Tactic: outline → full article
# ---------------------------------------------------------------------------

class WritingPipeline(Tactic):
    """
    Two-agent writing pipeline:
      1. 'outliner'  — produces a short bullet-point outline
      2. 'writer'    — expands the outline into a coherent article
    """

    name = "writing_pipeline"
    agent_group = ["outliner", "writer"]

    def call(self, task: str) -> str:
        outliner = self.agents["outliner"]
        writer = self.agents["writer"]

        # Stage 1: create an outline
        outliner.open("outline")
        outliner.receive(f"Create a concise 5-point outline for a short article about: {task}")
        outline = outliner.respond().content

        print("--- Outline ---")
        print(outline)
        print()

        # Stage 2: expand outline into an article
        writer.open("write")
        writer.receive(
            f"Write a short, engaging article (3-4 paragraphs) based on this outline:\n\n{outline}"
        )
        article = writer.respond().content

        return article


# ---------------------------------------------------------------------------
# Build tactic from config dict
# ---------------------------------------------------------------------------

config = {
    "global": {
        "model_name": MODEL,
        "model_args": {"temperature": 0.7},
    },
    "agent_configs": [
        {
            "name": "outliner",
            "system_prompt": (
                "You are an expert at creating concise, well-structured article outlines. "
                "Return bullet points only — no prose."
            ),
        },
        {
            "name": "writer",
            "system_prompt": (
                "You are a skilled writer who turns outlines into engaging, "
                "informative articles. Write clearly and avoid jargon."
            ),
            "model_args": {"temperature": 0.8},
        },
    ],
}

# noop_store() silences the 'no LogStore configured' warning
tactic = WritingPipeline(config, log_store=noop_store())

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

topic = "the impact of large language models on software development"
print(f"Topic: {topic}\n")

article = tactic(topic)

print("--- Article ---")
print(article)

# Inspect session cost
session = tactic(topic, return_session=True)
print(f"\nTotal cost: {session.total_cost}")
print(f"Agent calls: {session.agent_call_count}")
