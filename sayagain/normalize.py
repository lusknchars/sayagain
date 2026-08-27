"""Cross-language normalisation of tool-call arguments.

The harness varies how a thing is said; it must not then punish the agent for
saying the answer back differently. `sexta-feira`, `Friday` and `Fri` are the
same day. This module is the only place that decides that, and it is
deliberately small: every entry here is a chance to hide a real failure or
invent a fake one.

Two traps it exists to handle:

- `mañana` is both *tomorrow* and *morning*; `morgen` is both *tomorrow* and
  *morning*. Which one is meant depends on the field being filled, not on the
  word, so normalisation is always field-aware.
- Stripping combining marks is right for `manhã` and catastrophic for
  `शुक्रवार`, where the marks are the letters. Only Latin accents are stripped.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from typing import Any

#: Argument names that hold a day.
DATE_FIELDS = frozenset({"date", "day", "weekday"})
#: Argument names that hold a time of day.
TIME_FIELDS = frozenset({"time", "hour"})

#: The only matrix locale that writes dates month-first.
MONTH_FIRST_LOCALES = frozenset({"en-US"})

#: Canonical weekday -> surface forms, already accent-free and lowercase.
WEEKDAYS: dict[str, set[str]] = {
    "monday": {"monday", "mon", "segunda", "seg", "lunes", "lun", "montag", "somvar", "सोमवार"},
    "tuesday": {
        "tuesday",
        "tue",
        "tues",
        "terca",
        "ter",
        "martes",
        "dienstag",
        "mangalvar",
        "मंगलवार",
    },
    "wednesday": {
        "wednesday",
        "wed",
        "quarta",
        "qua",
        "miercoles",
        "mie",
        "mittwoch",
        "budhvar",
        "बुधवार",
    },
    "thursday": {
        "thursday",
        "thu",
        "thur",
        "thurs",
        "quinta",
        "qui",
        "jueves",
        "jue",
        "donnerstag",
        "guruvar",
        "गुरुवार",
    },
    "friday": {
        "friday",
        "fri",
        "fr",
        "sexta",
        "sex",
        "viernes",
        "vie",
        "freitag",
        "shukravar",
        "शुक्रवार",
    },
    "saturday": {"saturday", "sat", "sabado", "sab", "samstag", "shanivar", "शनिवार"},
    "sunday": {"sunday", "sun", "domingo", "dom", "sonntag", "ravivar", "रविवार"},
}

#: Canonical part of day -> surface forms.
DAYPARTS: dict[str, set[str]] = {
    "morning": {"morning", "manha", "manana", "morgen", "vormittag", "subah", "सुबह"},
    "afternoon": {"afternoon", "tarde", "nachmittag", "dopahar", "दोपहर"},
    "evening": {"evening", "night", "noite", "noche", "abend", "nacht", "sham", "शाम", "रात"},
}

#: Relative day -> surface forms, resolved against `today`.
RELATIVE_DAYS: dict[int, set[str]] = {
    0: {"today", "hoje", "hoy", "heute", "aaj", "आज"},
    1: {"tomorrow", "amanha", "manana", "morgen", "kal", "कल"},
    -1: {"yesterday", "ontem", "ayer", "gestern"},
}

# German two-letter weekday abbreviations (mo, di, mi, do, fr, sa, so) are
# deliberately absent except `fr`: "so" and "do" are ordinary English words and
# would turn every disfluent utterance into a Sunday.

_WEEKDAY_BY_FORM = {form: day for day, forms in WEEKDAYS.items() for form in forms}
_DAYPART_BY_FORM = {form: part for part, forms in DAYPARTS.items() for form in forms}
_OFFSET_BY_FORM = {form: offset for offset, forms in RELATIVE_DAYS.items() for form in forms}

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NUMERIC_DATE = re.compile(r"^(\d{1,2})[/.](\d{1,2})(?:[/.](\d{2,4}))?$")
_CLOCK = re.compile(r"^(\d{1,2}):(\d{2})$")

_LATIN_MAX = 0x0250


def normalise_text(text: str) -> str:
    """Lowercase, strip Latin accents, and collapse whitespace.

    Combining marks are dropped only when they sit on a Latin base character,
    so Devanagari and other scripts that build letters out of marks survive.
    """
    lowered = text.lower()
    out: list[str] = []
    on_latin = False
    for char in unicodedata.normalize("NFD", lowered):
        if unicodedata.combining(char):
            if not on_latin:
                out.append(char)
            continue
        on_latin = ord(char) < _LATIN_MAX
        out.append(char)
    return " ".join(unicodedata.normalize("NFC", "".join(out)).split())


def tokens(text: str) -> list[str]:
    r"""Split normalised text into words, keeping non-Latin scripts intact.

    Not a `\w+` regex: Python classifies Devanagari vowel signs as marks, not
    word characters, so `\w+` shreds `शुक्रवार` into fragments that match nothing.
    """
    words: list[str] = []
    current: list[str] = []
    for char in normalise_text(text):
        if char.isalnum() or unicodedata.category(char).startswith("M"):
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def normalise_value(value: Any, *, field: str, language: str, today: date | None = None) -> str:
    """Reduce one argument value to a canonical form for comparison."""
    text = value if isinstance(value, str) else str(value)
    if field in DATE_FIELDS:
        return _normalise_date(text, language=language, today=today or date.today())
    if field in TIME_FIELDS:
        return _normalise_time(text)
    return normalise_text(text)


def arguments_match(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    language: str,
    today: date | None = None,
) -> bool:
    """Report whether the agent's arguments mean what the scenario expected.

    Arguments the scenario does not mention are ignored, so an agent may return
    more than it was asked for.
    """
    for field, wanted in expected.items():
        if field not in actual:
            return False
        got = normalise_value(actual[field], field=field, language=language, today=today)
        want = normalise_value(wanted, field=field, language=language, today=today)
        if got != want:
            return False
    return True


def _normalise_date(text: str, *, language: str, today: date) -> str:
    stripped = normalise_text(text)
    if _ISO_DATE.match(stripped):
        return stripped

    numeric = _NUMERIC_DATE.match(stripped)
    if numeric:
        return _resolve_numeric_date(numeric, language=language, today=today)

    for token in tokens(stripped):
        if token in _WEEKDAY_BY_FORM:
            return _WEEKDAY_BY_FORM[token]

    for token in tokens(stripped):
        if token in _OFFSET_BY_FORM:
            return (today + timedelta(days=_OFFSET_BY_FORM[token])).isoformat()

    return stripped


def _resolve_numeric_date(match: re.Match[str], *, language: str, today: date) -> str:
    first, second = int(match.group(1)), int(match.group(2))
    year = _resolve_year(match.group(3), today)

    if language in MONTH_FIRST_LOCALES:
        month, day = first, second
    else:
        day, month = first, second

    if not 1 <= month <= 12:  # the other reading is the only one that can be right
        day, month = month, day
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return normalise_text(match.group(0))


def _resolve_year(raw: str | None, today: date) -> int:
    if raw is None:
        return today.year
    year = int(raw)
    return year + 2000 if year < 100 else year


def _normalise_time(text: str) -> str:
    stripped = normalise_text(text)
    clock = _CLOCK.match(stripped)
    if clock:
        return f"{int(clock.group(1)):02d}:{clock.group(2)}"

    for token in tokens(stripped):
        if token in _DAYPART_BY_FORM:
            return _DAYPART_BY_FORM[token]
    return stripped
