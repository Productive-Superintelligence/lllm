"""Serve the project's tactic over HTTP: /run, /stream, and /info."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from lllm.services import create_tactic_app  # noqa: E402

from main import create_tactic  # noqa: E402


def create_app():
    return create_tactic_app(create_tactic(), title="{{project_title}}")


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
