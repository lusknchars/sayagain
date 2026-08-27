"""Tests for expanding a scenario into the case matrix.

These build their own scenario rather than leaning on the bundled examples: the
examples exist to be edited, and a unit test that breaks when a translation is
added is testing the wrong thing.
"""

from pathlib import Path
from typing import Any

import yaml

from sayagain.expand import expand
from sayagain.scenario import Scenario, load_scenario


def partial_coverage(tmp_path: Path) -> Scenario:
    """Three languages, three registers, deliberately ragged coverage.

    formal covers all three languages, casual two, codeswitch one: six
    register+language pairs out of a possible nine.
    """
    data: dict[str, Any] = {
        "id": "partial",
        "description": "A scenario whose registers do not all cover every language.",
        "agent": {
            "system_prompt": "You are a test agent.",
            "tools": [{"name": "do_thing", "schema": {"date": "string"}}],
        },
        "matrix": {
            "language": ["en-US", "pt-BR", "de-DE"],
            "perturbation": ["clean", "telephone"],
        },
        "repeats": 3,
        "turns": [
            {
                "user": {
                    "intent": "do_the_thing",
                    "variants": {
                        "formal": {
                            "en-US": "Please do the thing on Friday.",
                            "pt-BR": "Por favor faça a coisa na sexta.",
                            "de-DE": "Bitte machen Sie das am Freitag.",
                        },
                        "casual": {
                            "en-US": "Do the thing Friday?",
                            "pt-BR": "Faz a coisa sexta?",
                        },
                        "codeswitch": {"en-US": "Do the thing na sexta?"},
                    },
                },
                "interrupt": {
                    "after_agent_speaks_ms": 500,
                    "with": {"en-US": "Friday, not Thursday.", "pt-BR": "Sexta, não quinta."},
                },
                "expect": {"tool_call": {"name": "do_thing", "arguments": {"date": "friday"}}},
            }
        ],
    }
    path = tmp_path / "partial.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_scenario(path)


def test_the_matrix_skips_languages_a_register_does_not_cover(tmp_path: Path) -> None:
    cases = expand(partial_coverage(tmp_path))

    # 6 register+language pairs x 2 perturbations x 3 repeats.
    assert len(cases) == 36
    assert {case.language for case in cases} == {"en-US", "pt-BR", "de-DE"}
    assert {case.register for case in cases if case.language == "de-DE"} == {"formal"}


def test_every_case_has_a_unique_id(tmp_path: Path) -> None:
    cases = expand(partial_coverage(tmp_path))

    assert len({case.id for case in cases}) == len(cases)


def test_a_case_carries_the_text_it_will_speak(tmp_path: Path) -> None:
    cases = expand(partial_coverage(tmp_path))

    case = next(c for c in cases if c.language == "pt-BR" and c.register == "casual")
    assert case.turn_texts == ["Faz a coisa sexta?"]
    assert case.interrupt_texts == ["Sexta, não quinta."]


def test_a_barge_in_with_no_line_in_this_language_is_dropped(tmp_path: Path) -> None:
    cases = expand(partial_coverage(tmp_path))

    case = next(c for c in cases if c.language == "de-DE")
    assert case.interrupt_texts == [None]


def test_repeats_are_numbered_from_one(tmp_path: Path) -> None:
    assert {case.repeat for case in expand(partial_coverage(tmp_path))} == {1, 2, 3}


def test_filtering_by_language(tmp_path: Path) -> None:
    cases = expand(partial_coverage(tmp_path), languages=["pt-BR"])

    assert {case.language for case in cases} == {"pt-BR"}
    assert len(cases) == 2 * 2 * 3  # formal and casual only


def test_filtering_by_perturbation(tmp_path: Path) -> None:
    cases = expand(partial_coverage(tmp_path), perturbations=["clean"])

    assert {case.perturbation.id for case in cases} == {"clean"}
    assert len(cases) == 6 * 3


def test_filtering_by_register(tmp_path: Path) -> None:
    cases = expand(partial_coverage(tmp_path), registers=["codeswitch"])

    assert {case.register for case in cases} == {"codeswitch"}
    assert len(cases) == 1 * 2 * 3


def test_overriding_repeats(tmp_path: Path) -> None:
    cases = expand(partial_coverage(tmp_path), repeats=1)

    assert len(cases) == 12
    assert {case.repeat for case in cases} == {1}


def test_a_filter_that_matches_nothing_yields_no_cases(tmp_path: Path) -> None:
    assert expand(partial_coverage(tmp_path), languages=["hi-IN"]) == []
