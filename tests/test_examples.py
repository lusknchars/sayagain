"""The bundled examples must stay loadable, expandable and honest."""

from pathlib import Path

import pytest

from sayagain.expand import expand
from sayagain.scenario import Scenario, load_scenarios

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCENARIOS = load_scenarios(EXAMPLES)
IDS = [scenario.id for scenario in SCENARIOS]


def test_there_are_three_examples() -> None:
    assert sorted(IDS) == ["order_status", "reschedule_appointment", "transfer_money"]


def test_scenario_ids_are_unique() -> None:
    assert len(set(IDS)) == len(IDS)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=IDS)
def test_every_example_expands_to_real_cases(scenario: Scenario) -> None:
    assert expand(scenario), f"{scenario.id} expands to nothing"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=IDS)
def test_every_example_has_its_prompt(scenario: Scenario) -> None:
    path = scenario.system_prompt_path()
    assert path is not None and path.is_file()
    assert scenario.system_prompt()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=IDS)
def test_no_language_in_the_matrix_is_dead_weight(scenario: Scenario) -> None:
    """A language nothing is ever said in silently shrinks the matrix."""
    covered = {
        language
        for turn in scenario.turns
        for by_language in turn.user.variants.values()
        for language in by_language
    }
    missing = sorted(set(scenario.matrix.language) - covered)
    assert not missing, f"{scenario.id} lists {missing} but never speaks them"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=IDS)
def test_no_translation_is_written_and_never_used(scenario: Scenario) -> None:
    assert scenario.unused_variant_languages() == []


@pytest.mark.parametrize("scenario", SCENARIOS, ids=IDS)
def test_every_register_covers_every_language(scenario: Scenario) -> None:
    """Partial registers are legal, but in the bundled examples they are a typo."""
    languages = set(scenario.matrix.language)
    for turn in scenario.turns:
        for register, by_language in turn.user.variants.items():
            missing = sorted(languages - set(by_language))
            assert not missing, f"{scenario.id}: register {register!r} is missing {missing}"


def test_the_locale_scenario_actually_depends_on_locale() -> None:
    """`transfer_money` is pointless unless 09/04 resolves differently per locale."""
    from datetime import date

    from sayagain.normalize import normalise_value

    scenario = next(s for s in SCENARIOS if s.id == "transfer_money")
    expected = scenario.turns[0].expect.tool_call
    assert expected is not None
    today = date(2026, 8, 27)

    readings = {
        language: normalise_value(
            expected.arguments["date"], field="date", language=language, today=today
        )
        for language in scenario.matrix.language
    }
    assert readings["en-US"] == "2026-09-04"
    assert readings["de-DE"] == "2026-04-09"
    assert len(set(readings.values())) == 2
