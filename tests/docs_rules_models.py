"""Canned model replies for docs/models.md (see docs_model.py)."""

from __future__ import annotations

from docs_model import asks, rule


@rule(asks("name one prime number above 90"))
def _prime(request):
    return "97"


@rule(asks("write a haiku about autumn rain"))
def _haiku(request):
    return "Cold rain on the eaves\nmaple leaves drift in puddles\nthe kettle whistles"


@rule(asks("say hello in spanish"))
def _hola(request):
    return "¡Hola!"


_flaky = {"calls": 0}


@rule(asks("boiling point of water at sea level"))
def _boiling(request):
    import litellm

    _flaky["calls"] += 1
    if _flaky["calls"] % 2:  # every first attempt loses its connection
        raise litellm.APIConnectionError(
            message="Connection reset by peer",
            llm_provider="openai",
            model=request.model,
        )
    return "100 °C, or 212 °F."


@rule(asks("summarize hamlet in one line"))
def _hamlet(request):
    return "A prince avenges his father's murder and destroys his court."
