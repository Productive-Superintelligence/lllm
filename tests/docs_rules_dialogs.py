"""Canned model replies for docs/dialogs.md (see docs_model.py)."""

from __future__ import annotations

from docs_model import asks, rule


def _user_texts(request) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in request.messages
        if message.get("role") == "user"
    ]


# -- Opening example ----------------------------------------------------------


@rule(asks("i have three days in lisbon"))
def _lisbon_days(request):
    return "Spend a day each on Alfama, Belém, and Sintra."


@rule(asks("what should i pack for may"))
def _lisbon_pack(request):
    return "Light layers, walking shoes, and a thin jacket."


# -- Messages -----------------------------------------------------------------


@rule(asks("suggest a name for a hiking app"))
def _hiking_name(request):
    return "Trailhead: short, outdoorsy, and hard to misspell."


# -- Forking --------------------------------------------------------------------


@rule(asks("checkout broke for 12 minutes", "payment api key expired"))
def _incident_notes(request):
    return "Noted: a 12-minute checkout outage caused by an expired API key."


@rule(asks("one-line status update for customers"))
def _incident_customers(request):
    return "Checkout was down 12 minutes; it works again."


@rule(asks("one-line note for the on-call channel"))
def _incident_engineers(request):
    return "Expired payment API key broke checkout for 12 min. Add expiry alerts."


# -- Saving and loading -------------------------------------------------------


@rule(asks("review this function name", "getdata2"))
def _review_name(request):
    return "Too vague. Try fetch_invoices or load_users."


# -- Images -----------------------------------------------------------------------


@rule(asks("what is the trend in this chart"))
def _chart_trend(request):
    has_image = any(
        isinstance(message.get("content"), list)
        and any(part.get("type") == "image_url" for part in message["content"])
        for message in request.messages
    )
    if not has_image:
        return "I don't see a chart."
    return "Revenue rises every quarter, with the biggest jump in Q4."


# -- Long conversations -------------------------------------------------------


@rule(asks("standup note:"))
def _standup_noted(request):
    return "Noted."


@rule(asks("what did i say on monday"))
def _standup_monday(request):
    earlier = _user_texts(request)[:-1]
    for text in earlier:
        if "monday" in text.lower():
            return text.split(":", 1)[1].strip()
    return "I no longer see Monday's note in our conversation."
