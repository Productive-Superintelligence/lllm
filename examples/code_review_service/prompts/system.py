"""
System prompts for the code review pipeline.

Both prompts are auto-registered by lllm via lllm.toml [prompts] discovery.
They are referenced by path in configs/default.yaml under system_prompt_path.
"""
from lllm.core.prompt import Prompt


analyzer_system = Prompt(
    path="system/analyzer",
    prompt=(
        "You are an expert code analyst. "
        "Your job is to carefully read code and identify issues: bugs, security problems, "
        "performance bottlenecks, anti-patterns, and style violations. "
        "Be specific — reference variable names and concrete code patterns. "
        "Do not suggest rewrites; only enumerate problems clearly."
    ),
    metadata={"role": "analyzer", "version": "1.0"},
)

synthesizer_system = Prompt(
    path="system/synthesizer",
    prompt=(
        "You are a senior code reviewer. "
        "Given a raw code analysis, you produce a clear, structured review "
        "that a developer can immediately act on. "
        "Be constructive and precise. Your output must follow the required JSON schema exactly."
    ),
    metadata={"role": "synthesizer", "version": "1.0"},
)
