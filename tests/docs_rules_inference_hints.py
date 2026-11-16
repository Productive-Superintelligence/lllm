"""Canned model replies for docs/inference-hints.md (see docs_model.py)."""

from __future__ import annotations

from docs_model import asks, rule


@rule(asks("plan a study of prefix caching"))
def _cache_plan(request):
    return (
        "1. Pick two workloads. 2. Measure time to first token with and "
        "without prefix caching. 3. Compare cache hit rates."
    )


@rule(asks("list the risks of the prefix caching study"))
def _cache_risks(request):
    return "Warm caches from earlier runs could hide the real cold-start cost."


@rule(asks("adapt the plan for a 1b model"))
def _cache_small(request):
    return "Use the 1B model with the same workloads and a larger batch size."
