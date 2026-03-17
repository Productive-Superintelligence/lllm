"""
Shared helper: detect which LLM provider API key is available.

Advanced examples import `MODEL` from here so you only need to set
one environment variable to run all of them:

    export OPENAI_API_KEY=sk-...
    # or
    export ANTHROPIC_API_KEY=sk-ant-...

You can also override by setting LLLM_EXAMPLE_MODEL directly:

    export LLLM_EXAMPLE_MODEL=gpt-4o-mini
"""
import os
import sys


def _detect_model() -> str:
    override = os.environ.get("LLLM_EXAMPLE_MODEL")
    if override:
        return override

    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o"

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-haiku-4-5-20251001"

    print(
        "No API key found.\n"
        "Set one of the following environment variables before running:\n"
        "  export OPENAI_API_KEY=sk-...\n"
        "  export ANTHROPIC_API_KEY=sk-ant-...\n"
        "Or specify the model directly:\n"
        "  export LLLM_EXAMPLE_MODEL=<model-id>",
        file=sys.stderr,
    )
    sys.exit(1)


MODEL: str = _detect_model()
