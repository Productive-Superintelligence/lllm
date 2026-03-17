"""
Structured output using Pydantic models.

Demonstrates:
  - Declaring a Pydantic model as the output schema (Prompt.format)
  - Accessing the parsed dict via message.parsed
  - Hydrating the dict back into the Pydantic model

Prerequisites:
    export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY, etc.

Note: structured output (response_format) requires a model that supports it,
e.g. gpt-4o, gpt-4o-mini, claude-3-5-sonnet, etc.
"""
from typing import List

from pydantic import BaseModel, Field

from lllm import Tactic, Prompt


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    rating: float = Field(description="Your rating out of 10")
    pros: List[str] = Field(description="Positive aspects (2-4 points)")
    cons: List[str] = Field(description="Negative aspects (1-3 points)")
    verdict: str = Field(description="One-sentence verdict")


# ---------------------------------------------------------------------------
# Prompt wired to the output schema
# ---------------------------------------------------------------------------

review_prompt = Prompt(
    path="review/movie",
    prompt=(
        "You are a film critic. "
        "Analyze the movie the user names and return a structured review "
        "that strictly follows the provided JSON schema."
    ),
    format=MovieReview,  # tells the LLM to respond with MovieReview-shaped JSON
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

agent = Tactic.quick(system_prompt=review_prompt)
agent.open("session")

agent.receive("Review the movie 'Inception' (2010) by Christopher Nolan.")
response = agent.respond()

# message.content is the raw JSON string
# message.parsed  is the dict from json.loads(content)
print("Raw JSON:\n", response.content[:200], "...\n")

# Hydrate into the Pydantic model for type-safe access
review = MovieReview(**response.parsed)
print(f"Title:   {review.title} ({review.year})")
print(f"Rating:  {review.rating}/10")
print(f"Pros:    {', '.join(review.pros)}")
print(f"Cons:    {', '.join(review.cons)}")
print(f"Verdict: {review.verdict}")
