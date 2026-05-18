from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("LLLM_CONFIG", str(PROJECT_ROOT / "lllm.toml"))
sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel  # noqa: E402

from main import build_app_tactic  # noqa: E402


class RunRequest(BaseModel):
    task: str


class RunResponse(BaseModel):
    result: str


def create_app():
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is not installed. Run `uv sync` or `pip install fastapi uvicorn`."
        ) from exc

    app = FastAPI(title="{{project_title}}")
    tactic = build_app_tactic()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/run", response_model=RunResponse)
    def run(request: RunRequest):
        return RunResponse(result=tactic(request.task))

    return app


app = create_app()


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Uvicorn is not installed. Run `uv sync` or `pip install uvicorn`."
        ) from exc

    uvicorn.run(app, host="0.0.0.0", port=8080)
