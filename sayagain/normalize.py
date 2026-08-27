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

#: Month number -> surface forms. Abbreviations that are ordinary words or that
#: collide across languages are deliberately absent: `mar` is short for March,
#: for Spanish `martes`, and for the sea, and guessing any of them is worse than
#: reporting no date at all.
MONTHS: dict[int, set[str]] = {
    1: {"january", "jan", "enero", "ene", "janeiro", "januar", "जनवरी"},
    2: {"february", "feb", "febrero", "fevereiro", "fev", "februar", "फरवरी"},
    3: {"march", "marzo", "marco", "marz", "मार्च"},
    4: {"april", "abril", "abr", "अप्रैल"},
    5: {"may", "mayo", "maio", "mai", "मई"},
    6: {"june", "jun", "junio", "junho", "juni", "जून"},
    7: {"july", "jul", "julio", "julho", "juli", "जुलाई"},
    8: {"august", "aug", "agosto", "ago", "अगस्त"},
    9: {"september", "sept", "sep", "septiembre", "setembro", "set", "सितंबर"},
    10: {"october", "oct", "octubre", "outubro", "out", "oktober", "okt", "अक्टूबर"},
    11: {"november", "nov", "noviembre", "novembro", "नवंबर"},
    12: {"december", "dec", "diciembre", "dic", "dezembro", "dez", "dezember", "दिसंबर"},
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
_MONTH_BY_FORM = {form: month for month, forms in MONTHS.items() for form in forms}

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NUMERIC_DATE = re.compile(r"^(\d{1,2})[/.](\d{1,2})(?:[/.](\d{2,4}))?$")
_NUMERIC_DATE_IN_TEXT = re.compile(r"\b(\d{1,2})[/.](\d{1,2})(?:[/.](\d{2,4}))?\b")
_ISO_DATE_IN_TEXT = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
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


def find_date(text: str, *, language: str, today: date | None = None) -> str | None:
    """Find a date anywhere in a sentence and reduce it to canonical form.

    Ordered deliberately: an explicit ISO date beats a numeric one, a numeric one
    beats a weekday, and a weekday beats a relative day, because that is the
    order of how specific they are. Returns None when the sentence names no date
    at all, which is different from naming one badly.
    """
    stripped = normalise_text(text)
    when = today or date.today()

    iso = _ISO_DATE_IN_TEXT.search(stripped)
    if iso:
        return iso.group(0)

    numeric = _NUMERIC_DATE_IN_TEXT.search(stripped)
    if numeric:
        return _resolve_numeric_date(numeric, language=language, today=when)

    spoken = _find_month_and_day(tokens(stripped), today=when)
    if spoken is not None:
        return spoken

    for token in tokens(stripped):
        if token in _WEEKDAY_BY_FORM:
            return _WEEKDAY_BY_FORM[token]

    for token in tokens(stripped):
        if token in _OFFSET_BY_FORM:
            return (when + timedelta(days=_OFFSET_BY_FORM[token])).isoformat()

    return None


def find_daypart(text: str) -> str | None:
    """Find a part of day anywhere in a sentence, or None if none is named."""
    for token in tokens(text):
        if token in _DAYPART_BY_FORM:
            return _DAYPART_BY_FORM[token]
    return None


def _normalise_date(text: str, *, language: str, today: date) -> str:
    stripped = normalise_text(text)
    if _ISO_DATE.match(stripped):
        return stripped
    numeric = _NUMERIC_DATE.match(stripped)
    if numeric:
        return _resolve_numeric_date(numeric, language=language, today=today)
    found = find_date(stripped, language=language, today=today)
    return found if found is not None else stripped


def _find_month_and_day(words: list[str], *, today: date) -> str | None:
    """Read a spoken date such as `September 4th`, `9. April` or `4 de abril`.

    Word order differs by language, so the day is looked for on either side of
    the month name, nearest first. A four-digit number nearby is read as a year,
    never as a day.
    """
    for index, word in enumerate(words):
        month = _MONTH_BY_FORM.get(word)
        if month is None:
            continue
        day = _nearby_number(words, index, low=1, high=31, digits=2)
        if day is None:
            continue
        year = _nearby_number(words, index, low=1900, high=2999, digits=4) or today.year
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None
    return None


def _nearby_number(words: list[str], index: int, *, low: int, high: int, digits: int) -> int | None:
    """Find a number beside `index`, closest first, within the given range."""
    for offset in (-1, 1, -2, 2):
        position = index + offset
        if not 0 <= position < len(words):
            continue
        word = _drop_ordinal_suffix(words[position])
        if not word.isdigit() or len(word) > digits:
            continue
        value = int(word)
        if low <= value <= high:
            return value
    return None


def _drop_ordinal_suffix(word: str) -> str:
    """Turn `4th`, `1st`, `2nd`, `3rd` into the bare number."""
    for suffix in ("st", "nd", "rd", "th"):
        if word.endswith(suffix) and word[: -len(suffix)].isdigit():
            return word[: -len(suffix)]
    return word


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
