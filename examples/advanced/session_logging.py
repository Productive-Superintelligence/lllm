"""
Session logging and querying with LogStore.

Demonstrates:
  - Attaching a SQLite-backed LogStore to a Tactic
  - Automatic session persistence after every tactic call
  - Listing sessions with list_sessions()
  - Loading a full session record and inspecting cost

Usage:
    python examples/advanced/session_logging.py

Prerequisites:
    export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY, etc.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from _api_key import MODEL  # noqa: E402

from lllm import Tactic
from lllm.logging import sqlite_store


# ---------------------------------------------------------------------------
# Custom Tactic
# ---------------------------------------------------------------------------

class Summarizer(Tactic):
    """Single-agent tactic that summarizes a piece of text."""

    name = "summarizer"
    agent_group = ["summarizer"]

    def call(self, task: str) -> str:
        agent = self.agents["summarizer"]
        agent.open("session")
        agent.receive(task)
        return agent.respond().content


config = {
    "global": {"model_name": MODEL},
    "agent_configs": [
        {
            "name": "summarizer",
            "system_prompt": (
                "You are a concise summarizer. "
                "Return a 1-2 sentence summary of whatever the user sends you."
            ),
        }
    ],
}

# ---------------------------------------------------------------------------
# Set up a persistent SQLite log store
# ---------------------------------------------------------------------------

DB_PATH = "/tmp/lllm_session_demo.db"
store = sqlite_store(DB_PATH, partition="summarizer_demo")

tactic = Summarizer(config, log_store=store)

# ---------------------------------------------------------------------------
# Run a few tasks — each call is automatically persisted
# ---------------------------------------------------------------------------

texts = [
    (
        "Python is a high-level, general-purpose programming language. "
        "Its design philosophy emphasises code readability using significant indentation."
    ),
    (
        "Machine learning is a subset of artificial intelligence that enables systems "
        "to learn and improve from experience without being explicitly programmed."
    ),
    (
        "The transformer architecture, introduced in 'Attention Is All You Need' (2017), "
        "revolutionised natural language processing by replacing recurrent networks "
        "with self-attention mechanisms."
    ),
]

print("Running tasks...\n")
for text in texts:
    summary = tactic(text, tags={"source": "demo"})
    print(f"Input : {text[:60]}...")
    print(f"Output: {summary}\n")

# ---------------------------------------------------------------------------
# Query the log store
# ---------------------------------------------------------------------------

summaries = store.list_sessions()
print(f"Stored {len(summaries)} session(s) in {DB_PATH}:\n")
for s in summaries:
    print(f"  id={s.session_id[:12]}  tactic={s.tactic_name}  state={s.state}")

# Load the last session for detailed inspection
if summaries:
    last = store.load_session_record(summaries[-1].session_id)
    print(f"\nLast session detail:")
    print(f"  Total cost : {last.session.total_cost}")
    print(f"  Agent calls: {last.session.agent_call_count}")
    print(f"  Delivery   : {str(last.session.delivery)[:80]}...")
