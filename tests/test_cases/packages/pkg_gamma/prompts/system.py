"""System prompts for pkg_gamma."""
from lllm.core.prompt import Prompt

analyst = Prompt(
    path="analyst",
    prompt="You are a data analyst. Analyze {dataset} and provide insights.",
)

visualizer = Prompt(
    path="visualizer",
    prompt="Create a visualization plan for the {chart_type} chart showing {metric}.",
)
