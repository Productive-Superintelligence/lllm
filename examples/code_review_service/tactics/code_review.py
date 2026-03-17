"""
CodeReviewTactic — two-agent code review pipeline.

Stage 1  analyzer   : reads the raw code, produces a free-form analysis.
Stage 2  synthesizer: converts the analysis into a structured CodeReviewResult.

The output model (CodeReviewResult) is stored as `review_task.format` in the
prompt registry, so the tactic accesses it without a direct cross-file import:

    review_prompt = load_prompt("task/review")
    OutputModel   = review_prompt.format
    result        = OutputModel(**response.parsed)

Config format (from configs/default.yaml or pro.yaml)::

    tactic_type: code_review
    global:
      model_name: gpt-4o-mini
    agent_configs:
      - name: analyzer
        system_prompt_path: system/analyzer
      - name: synthesizer
        system_prompt_path: system/synthesizer
"""
from pydantic import BaseModel

from lllm import Tactic, load_prompt


class CodeInput(BaseModel):
    """Input to CodeReviewTactic.call()."""
    code: str
    language: str = "python"


class CodeReviewTactic(Tactic):
    """
    Two-stage code review pipeline.

    Agents declared in agent_group must match the names in agent_configs
    inside the config dict passed to __init__.
    """

    name = "code_review"
    agent_group = ["analyzer", "synthesizer"]

    def call(self, task: CodeInput):
        analyzer    = self.agents["analyzer"]
        synthesizer = self.agents["synthesizer"]

        # Load task prompts from the runtime registry (registered via lllm.toml)
        analyze_prompt = load_prompt("task/analyze")
        review_prompt  = load_prompt("task/review")

        # ── Stage 1: free-form code analysis ────────────────────────────────
        analyzer.open("analysis")
        analyzer.receive_prompt(
            analyze_prompt,
            {"code": task.code, "language": task.language},
        )
        analysis: str = analyzer.respond().content

        # ── Stage 2: structured review ───────────────────────────────────────
        # review_prompt.format == CodeReviewResult (set in prompts/tasks.py)
        # The LLM is instructed to return JSON that matches that schema.
        synthesizer.open("review")
        synthesizer.receive_prompt(review_prompt, {"analysis": analysis})
        response = synthesizer.respond()

        OutputModel = review_prompt.format   # CodeReviewResult class
        return OutputModel(**response.parsed)
