"""Tests for cross-language normalisation of tool-call arguments."""

from datetime import date

import pytest

from sayagain.normalize import arguments_match, normalise_text, normalise_value

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
