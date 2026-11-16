"""Canned model replies for docs/pydantic-ai.md (see docs_model.py).

Pydantic AI agents on that page run on Pydantic AI's own TestModel and
FunctionModel; these rules answer the native LLLM agents that work with them.
"""

from __future__ import annotations

import json

from docs_model import asks, rule


@rule(asks("rate the risk of shipping this change", "database driver"))
def _release_risk(request):
    return json.dumps(
        {
            "level": "high",
            "reason": "Every query goes through the driver.",
        }
    )
