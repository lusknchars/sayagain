"""Tests for expanding a scenario into the case matrix."""

from pathlib import Path

from sayagain.expand import expand
from sayagain.scenario import load_scenario

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "reschedule_appointment.yaml"


def test_the_matrix_skips_languages_a_register_does_not_cover() -> None:
    scenario = load_scenario(EXAMPLE)

    cases = expand(scenario)

    # en-US has 4 registers, pt-BR has 3 (no codeswitch), the other four have none.
    # 7 pairs x 6 perturbations x 3 repeats.
    assert len(cases) == 126
    assert {case.language for case in cases} == {"en-US", "pt-BR"}


def test_every_case_has_a_unique_id() -> None:
    cases = expand(load_scenario(EXAMPLE))

    assert len({case.id for case in cases}) == len(cases)


def test_a_case_carries_the_text_it_will_speak() -> None:
    cases = expand(load_scenario(EXAMPLE))

    case = next(c for c in cases if c.language == "pt-BR" and c.register == "casual")
    assert case.turn_texts == ["Dá pra jogar minha consulta pra sexta de manhã?"]
    assert case.interrupt_texts == ["Desculpa — sexta, não quinta."]


def test_repeats_are_numbered_from_one() -> None:
    cases = expand(load_scenario(EXAMPLE))

    repeats = {case.repeat for case in cases}
    assert repeats == {1, 2, 3}


def test_filtering_by_language() -> None:
    cases = expand(load_scenario(EXAMPLE), languages=["pt-BR"])

    assert {case.language for case in cases} == {"pt-BR"}
    assert len(cases) == 3 * 6 * 3


def test_filtering_by_perturbation() -> None:
    cases = expand(load_scenario(EXAMPLE), perturbations=["clean"])

    assert {case.perturbation.id for case in cases} == {"clean"}
    assert len(cases) == 7 * 3


def test_overriding_repeats() -> None:
    cases = expand(load_scenario(EXAMPLE), repeats=1)

    assert len(cases) == 42
    assert {case.repeat for case in cases} == {1}


def test_a_filter_that_matches_nothing_yields_no_cases() -> None:
    assert expand(load_scenario(EXAMPLE), languages=["hi-IN"]) == []
