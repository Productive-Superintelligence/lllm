"""Canned model replies for docs/streaming.md (see docs_model.py)."""

from __future__ import annotations

from docs_model import asks, rule


@rule(asks("list three facts about tide pools"))
def _tide_facts(request):
    return (
        "1. Tide pools form where rock holds seawater at low tide.\n"
        "2. Anemones, crabs, and snails live in them.\n"
        "3. Their temperature and salinity swing every day."
    )


@rule(asks("write a one-line brief from these notes", "tide pools"))
def _tide_brief(request):
    return "Tide pools shelter hardy sea life between tides."


@rule(asks("two-sentence story about a lighthouse"))
def _lighthouse(request):
    return "The keeper lit the lamp one last time. At dawn, a ship waved back."


@rule(asks("write one line about the sea"))
def _sea(request):
    return "The sea keeps every secret it is told."
