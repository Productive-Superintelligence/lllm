"""
Code Review Service
===================
Wraps CodeReviewTactic as a FastAPI HTTP service.

Package layout
--------------
    code_review_service/
    ├── lllm.toml          ← auto-discovered by lllm on startup
    ├── service.py         ← this file (entry point)
    ├── prompts/
    │   ├── system.py      ← system prompts (auto-registered)
    │   └── tasks.py       ← task prompts + CodeReviewResult schema
    ├── tactics/
    │   └── code_review.py ← CodeReviewTactic (auto-registered)
    └── configs/
        ├── default.yaml   ← base config (fast model)
        └── pro.yaml       ← production config (higher-quality model)

Usage
-----
Install deps (web server only):
    pip install fastapi uvicorn

Start the web service:
    cd examples/code_review_service
    python service.py

Run a quick demo without a web server:
    python service.py --demo

Use the production config:
    LLLM_CONFIG_PROFILE=pro python service.py

Override the model (any LiteLLM model ID):
    LLLM_EXAMPLE_MODEL=gpt-4o-mini python service.py --demo

Test the running service:
    curl -X POST http://localhost:8080/review \\
         -H 'Content-Type: application/json' \\
         -d '{"code": "def add(a,b): return a+b", "language": "python"}'
"""
import os
import sys

# ── Bootstrap: must happen before `import lllm` ──────────────────────────────
#
# 1. Point lllm.toml discovery at THIS package directory so that
#    `import lllm` finds the right config regardless of where the user
#    calls `python service.py` from.
#
# 2. Add the package root to sys.path so that
#    `from tactics.code_review import CodeInput` works as a normal import.

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

if "LLLM_CONFIG" not in os.environ:
    os.environ["LLLM_CONFIG"] = os.path.join(_here, "lllm.toml")

# ── API-key / model detection ─────────────────────────────────────────────────

def _detect_model() -> str:
    override = os.environ.get("LLLM_EXAMPLE_MODEL")
    if override:
        return override
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-haiku-4-5-20251001"
    print(
        "No API key found.\n"
        "Set one of the following before running:\n"
        "  export OPENAI_API_KEY=sk-...\n"
        "  export ANTHROPIC_API_KEY=sk-ant-...\n"
        "Or pin a specific model:\n"
        "  export LLLM_EXAMPLE_MODEL=gpt-4o-mini",
        file=sys.stderr,
    )
    sys.exit(1)


MODEL = _detect_model()

# ── LLLM setup ────────────────────────────────────────────────────────────────
#
# `import lllm` triggers _auto_init() which calls load_runtime().
# load_runtime() honours LLLM_CONFIG, so it finds our package's lllm.toml
# and auto-registers all prompts, tactics, and configs inside it.

import lllm  # noqa: E402  (must come after env-var setup above)
from lllm import build_tactic
from lllm.core.config import resolve_config
from lllm.logging import sqlite_store

# Select config profile: "default" (fast) or "pro" (higher-quality)
_profile = os.environ.get("LLLM_CONFIG_PROFILE", "default")
_config = resolve_config(_profile)

# Inject the auto-detected model into the global config so that all agents
# pick it up.  Per-agent model_name (if any) would override this via
# deep-merge in parse_agent_configs.
_config.setdefault("global", {})["model_name"] = MODEL

# Persist sessions to a SQLite file next to this script
_log_store = sqlite_store(os.path.join(_here, "sessions.db"))

# Build the tactic once at startup; each call() creates fresh agent copies
tactic = build_tactic(_config, log_store=_log_store)

# Import CodeInput after sys.path is set up (allows `from tactics.code_review import`)
from tactics.code_review import CodeInput  # noqa: E402

# ── Demo mode ─────────────────────────────────────────────────────────────────

_DEMO_CODE = """\
def find_duplicates(lst):
    seen = []
    duplicates = []
    for item in lst:
        if item in seen:
            duplicates.append(item)
        seen.append(item)
    return duplicates
"""

_DEMO_LANGUAGE = "python"


def run_demo() -> None:
    task = CodeInput(code=_DEMO_CODE, language=_DEMO_LANGUAGE)

    print(f"Model   : {MODEL}")
    print(f"Profile : {_profile}")
    print("\n--- Input code ---")
    print(_DEMO_CODE)

    print("Reviewing… (this may take a few seconds)\n")
    result = tactic(task)

    print("--- Review ---")
    print(f"Rating      : {result.rating}/10")
    print(f"Summary     : {result.summary}")
    print(f"\nIssues ({len(result.issues)}):")
    for issue in result.issues:
        print(f"  • {issue}")
    print(f"\nSuggestions ({len(result.suggestions)}):")
    for suggestion in result.suggestions:
        print(f"  • {suggestion}")

    # Show diagnostics via return_session=True
    session = tactic(task, return_session=True)
    print(f"\nTotal cost  : {session.total_cost}")
    print(f"Agent calls : {session.agent_call_count}")
    print(f"Sessions DB : {os.path.join(_here, 'sessions.db')}")


# ── FastAPI app ───────────────────────────────────────────────────────────────

def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel as _BM
        from typing import List as _List
    except ImportError:
        print(
            "FastAPI is not installed.\n"
            "Run:  pip install fastapi uvicorn\n"
            "Or test without a web server:  python service.py --demo",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Request / Response models (independent of the internal Pydantic models)

    class ReviewRequest(_BM):
        code: str
        language: str = "python"

    class ReviewResponse(_BM):
        summary: str
        issues: _List[str]
        suggestions: _List[str]
        rating: int

    # ── App ──────────────────────────────────────────────────────────────────

    app = FastAPI(
        title="Code Review Service",
        description=(
            "Two-agent pipeline: an analyzer reads the code and an synthesizer "
            "produces a structured review with issues, suggestions, and a rating."
        ),
        version="0.1.0",
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL, "profile": _profile}

    @app.post("/review", response_model=ReviewResponse)
    def review(req: ReviewRequest):
        try:
            task = CodeInput(code=req.code, language=req.language)
            result = tactic(task)
            return ReviewResponse(
                summary=result.summary,
                issues=result.issues,
                suggestions=result.suggestions,
                rating=result.rating,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        try:
            import uvicorn
        except ImportError:
            print(
                "uvicorn is not installed.\n"
                "Run:  pip install fastapi uvicorn\n"
                "Or test without a web server:  python service.py --demo",
                file=sys.stderr,
            )
            sys.exit(1)
        app = create_app()
        print(f"Starting Code Review Service on http://0.0.0.0:8080  (model={MODEL}, profile={_profile})")
        uvicorn.run(app, host="0.0.0.0", port=8080)
