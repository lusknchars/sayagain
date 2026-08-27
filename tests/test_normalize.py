"""Tests for cross-language normalisation of tool-call arguments."""

from datetime import date

import pytest

from sayagain.normalize import (
    arguments_match,
    find_date,
    find_daypart,
    normalise_text,
    normalise_value,
)

TODAY = date(2026, 8, 26)  # a Wednesday, fixed so relative dates are testable


@pytest.mark.parametrize(
    ("spoken", "language"),
    [
        ("Friday", "en-US"),
        ("Friday", "en-IN"),
        ("viernes", "es-MX"),
        ("sexta-feira", "pt-BR"),
        ("Freitag", "de-DE"),
        ("शुक्रवार", "hi-IN"),
    ],
)
def test_friday_in_every_matrix_language(spoken: str, language: str) -> None:
    assert normalise_value(spoken, field="date", language=language, today=TODAY) == "friday"


@pytest.mark.parametrize("spoken", ["Fri", "fri.", "sex", "vie", "Fr"])
def test_common_weekday_abbreviations(spoken: str) -> None:
    assert normalise_value(spoken, field="date", language="en-US", today=TODAY) == "friday"


@pytest.mark.parametrize(
    ("spoken", "language"),
    [
        ("morning", "en-US"),
        ("in the morning", "en-US"),
        ("de manhã", "pt-BR"),
        ("por la mañana", "es-MX"),
        ("am Morgen", "de-DE"),
        ("सुबह", "hi-IN"),
    ],
)
def test_morning_in_every_matrix_language(spoken: str, language: str) -> None:
    assert normalise_value(spoken, field="time", language=language, today=TODAY) == "morning"


def test_manana_means_tomorrow_as_a_date_and_morning_as_a_time() -> None:
    assert normalise_value("mañana", field="date", language="es-MX", today=TODAY) == "2026-08-27"
    assert normalise_value("mañana", field="time", language="es-MX", today=TODAY) == "morning"


def test_german_morgen_is_the_same_trap() -> None:
    assert normalise_value("morgen", field="date", language="de-DE", today=TODAY) == "2026-08-27"
    assert normalise_value("morgen", field="time", language="de-DE", today=TODAY) == "morning"


def test_devanagari_survives_normalisation() -> None:
    # Stripping combining marks the way you would for Latin accents destroys Hindi.
    assert normalise_text("शुक्रवार") == "शुक्रवार"


def test_latin_accents_are_still_stripped() -> None:
    assert normalise_text("sexta-feira de manhã") == "sexta-feira de manha"


def test_the_ninth_of_april_is_not_the_fourth_of_september_everywhere() -> None:
    american = normalise_value("09/04", field="date", language="en-US", today=TODAY)
    brazilian = normalise_value("09/04", field="date", language="pt-BR", today=TODAY)
    german = normalise_value("09/04", field="date", language="de-DE", today=TODAY)

    assert american == "2026-09-04"
    assert brazilian == "2026-04-09"
    assert german == "2026-04-09"


def test_iso_dates_pass_straight_through() -> None:
    assert (
        normalise_value("2026-09-04", field="date", language="pt-BR", today=TODAY) == "2026-09-04"
    )


def test_clock_times_are_kept() -> None:
    assert normalise_value("09:30", field="time", language="en-US", today=TODAY) == "09:30"


def test_next_friday_is_still_friday() -> None:
    assert normalise_value("next Friday", field="date", language="en-US", today=TODAY) == "friday"


def test_unknown_fields_fall_back_to_plain_text() -> None:
    assert (
        normalise_value("  Dr. SILVA ", field="doctor", language="pt-BR", today=TODAY)
        == "dr. silva"
    )


# --- the API the scorer actually calls ------------------------------------


def test_arguments_match_across_languages() -> None:
    expected = {"date": "friday", "time": "morning"}
    actual = {"date": "sexta-feira", "time": "de manhã"}

    assert arguments_match(expected, actual, language="pt-BR", today=TODAY) is True


def test_arguments_do_not_match_on_the_wrong_day() -> None:
    expected = {"date": "friday", "time": "morning"}
    actual = {"date": "quinta-feira", "time": "de manhã"}

    assert arguments_match(expected, actual, language="pt-BR", today=TODAY) is False


def test_extra_arguments_from_the_agent_are_ignored() -> None:
    expected = {"date": "friday"}
    actual = {"date": "Friday", "confirmation_sent": True}

    assert arguments_match(expected, actual, language="en-US", today=TODAY) is True


def test_a_missing_argument_fails() -> None:
    assert arguments_match({"date": "friday"}, {}, language="en-US", today=TODAY) is False


def test_matching_is_not_confused_by_case_or_punctuation() -> None:
    assert arguments_match(
        {"date": "friday"}, {"date": "  FRIDAY!  "}, language="en-US", today=TODAY
    )


# --- finding a date inside a sentence --------------------------------------


def test_find_date_picks_a_weekday_out_of_a_sentence() -> None:
    found = find_date("I would like to move it to Friday morning", language="en-US", today=TODAY)

    assert found == "friday"


def test_find_date_reads_a_numeric_date_by_locale() -> None:
    sentence_us = find_date("please move it to 09/04", language="en-US", today=TODAY)
    sentence_de = find_date("verschieben Sie es auf 09/04", language="de-DE", today=TODAY)

    assert sentence_us == "2026-09-04"
    assert sentence_de == "2026-04-09"


def test_find_date_prefers_an_explicit_iso_date() -> None:
    assert find_date("book it for 2026-12-01 please", language="pt-BR", today=TODAY) == "2026-12-01"


def test_find_date_understands_relative_days() -> None:
    assert find_date("pode ser amanhã?", language="pt-BR", today=TODAY) == "2026-08-27"


def test_find_date_returns_nothing_when_there_is_no_date() -> None:
    assert find_date("can you help me please", language="en-US", today=TODAY) is None


def test_find_date_takes_the_first_weekday_because_corrections_come_first() -> None:
    found = find_date("Friday, not Thursday", language="en-US", today=TODAY)

    assert found == "friday"


def test_find_daypart_picks_a_part_of_day_out_of_a_sentence() -> None:
    assert find_daypart("pode ser de manhã?") == "morning"
    assert find_daypart("irgendwann am Nachmittag") == "afternoon"
    assert find_daypart("शुक्रवार सुबह") == "morning"


def test_find_daypart_returns_nothing_when_unstated() -> None:
    assert find_daypart("move it to Friday") is None


# --- spoken month-and-day dates -------------------------------------------
#
# A voice harness never sees "09/04": text-to-speech reads it aloud in its own
# locale, so the agent hears "September 4th" or "9. April". Reading those back
# is therefore not a nicety, it is the whole locale test.


@pytest.mark.parametrize(
    ("sentence", "language", "expected"),
    [
        ("Please transfer 200 on September 4.", "en-US", "2026-09-04"),
        ("Can you send 200 on September 4th?", "en-US", "2026-09-04"),
        ("Bitte überweisen Sie 200 am 9. April.", "de-DE", "2026-04-09"),
        ("Kannst du am 9. April 200 überweisen?", "de-DE", "2026-04-09"),
        ("Por favor transfiera 200 el 4 de septiembre.", "es-MX", "2026-09-04"),
        ("Por favor, transferir 200 no dia 9 de abril.", "pt-BR", "2026-04-09"),
        ("कृपया 9 अप्रैल को 200 ट्रांसफर करें।", "hi-IN", "2026-04-09"),
    ],
)
def test_spoken_dates_resolve_to_the_same_day_a_locale_means(
    sentence: str, language: str, expected: str
) -> None:
    assert find_date(sentence, language=language, today=TODAY) == expected


def test_a_month_and_day_beats_a_weekday_in_the_same_sentence() -> None:
    found = find_date("Friday the 4th of September works", language="en-US", today=TODAY)

    assert found == "2026-09-04"


def test_a_year_in_the_sentence_is_not_mistaken_for_a_day() -> None:
    assert find_date("on 4 September 2027", language="en-US", today=TODAY) == "2027-09-04"


def test_a_month_with_no_day_is_not_a_date() -> None:
    assert find_date("sometime in September maybe", language="en-US", today=TODAY) is None


def test_an_ambiguous_abbreviation_is_not_guessed_at() -> None:
    # "mar" is short for March, for Spanish `martes`, and is also the sea.
    # Reporting no date is better than picking one of three readings.
    assert find_date("nos vemos el mar", language="es-MX", today=TODAY) is None
