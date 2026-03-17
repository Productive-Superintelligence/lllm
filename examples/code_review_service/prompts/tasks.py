"""
Task prompts for the code review pipeline.

  analyze_task  — sent to the analyzer agent with the raw code
  review_task   — sent to the synthesizer agent with the analysis text;
                  format=CodeReviewResult instructs the LLM to return
                  structured JSON matching the schema.

CodeReviewResult is defined here (not in a separate models file) so that
tactics/code_review.py can access it cleanly via:

    review_prompt = load_prompt("task/review")
    OutputModel   = review_prompt.format          # → CodeReviewResult class
    result        = OutputModel(**response.parsed)
"""
from typing import List

from pydantic import BaseModel, Field

from lllm.core.prompt import Prompt


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class CodeReviewResult(BaseModel):
    """Structured output produced by the synthesizer agent."""
    summary: str = Field(
        description="2-3 sentence overall assessment of the code quality."
    )
    issues: List[str] = Field(
        description=(
            "Specific problems found: bugs, security issues, anti-patterns, "
            "or performance concerns. Each entry is one concrete issue."
        )
    )
    suggestions: List[str] = Field(
        description="Actionable improvements the developer should make."
    )
    rating: int = Field(
        description=(
            "Overall code quality score: 1 (very poor) to 10 (excellent). "
            "7+ means production-ready with minor notes."
        )
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Stage 1 — raw analysis (free-form text)
analyze_task = Prompt(
    path="task/analyze",
    prompt=(
        "Analyze the following {language} code.\n"
        "Identify all bugs, anti-patterns, security issues, and improvement opportunities.\n"
        "Be thorough and specific — name the exact variables, functions, and patterns involved.\n\n"
        "```{language}\n{code}\n```"
    ),
    metadata={"stage": "1-analyze"},
)

# Stage 2 — structured review (LLM must respond with CodeReviewResult JSON)
review_task = Prompt(
    path="task/review",
    prompt=(
        "Based on the code analysis below, produce a structured review.\n\n"
        "Analysis:\n{analysis}"
    ),
    format=CodeReviewResult,
    metadata={"stage": "2-review"},
)
