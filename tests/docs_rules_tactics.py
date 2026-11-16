"""Canned model replies for docs/tactics.md (see docs_model.py)."""

from __future__ import annotations

import json

from docs_model import asks, rule

# -- Plan, research in parallel, write ----------------------------------------

HEAT_PUMP_ANSWERS = {
    "How does a heat pump move heat?": (
        "It moves heat with a refrigerant cycle instead of burning fuel, "
        "so it delivers more energy than it uses."
    ),
    "Do heat pumps work in cold climates?": (
        "Cold-climate models keep heating well below freezing, "
        "down to about -25°C."
    ),
    "What does a heat pump cost to run?": (
        "Each unit of electricity gives 2.5 to 4 units of heat, "
        "so running costs depend on power prices."
    ),
}


@rule(asks("list three questions to research about", "heat pumps"))
def _plan_heat_pumps(request):
    return json.dumps({"questions": list(HEAT_PUMP_ANSWERS)})


def _question(request) -> str:
    return request.text.removeprefix("Answer in one sentence: ").strip()


@rule(
    lambda r: r.text.startswith("Answer in one sentence: ")
    and _question(r) in HEAT_PUMP_ANSWERS
)
def _answer_question(request):
    return HEAT_PUMP_ANSWERS[_question(request)]


@rule(asks("write a report on heat pumps", "one point per finding"))
def _report_heat_pumps(request):
    return json.dumps(
        {
            "title": "Heat pumps in three points",
            "points": [
                "They move heat instead of burning fuel, so they give more "
                "than they use.",
                "Cold-climate models keep heating down to about -25°C.",
                "Each unit of electricity yields 2.5 to 4 units of heat.",
            ],
        }
    )


# -- Fresh agents for every call ----------------------------------------------


@rule(asks("remember this code: 4417"))
def _remember_code(request):
    return "Got it: 4417."


@rule(asks("what code did i give you"))
def _recall_code(request):
    earlier = [
        str(message.get("content") or "")
        for message in request.messages[:-1]
        if message.get("role") == "user"
    ]
    if any("4417" in text for text in earlier):
        return "You gave me 4417."
    return "You haven't given me a code yet."


# -- Calling a tactic ---------------------------------------------------------


@rule(asks("label this product review as positive or negative"))
def _label_review(request):
    review = request.text.split(":", 1)[1].lower()
    negative = ("broke", "late", "refund", "stopped", "never")
    return "negative" if any(word in review for word in negative) else "positive"


# -- One agent: Tactic.quick ----------------------------------------------------


@rule(asks("write a commit message for", "fix typo in readme"))
def _commit_message(request):
    return "docs: fix typo in README"
