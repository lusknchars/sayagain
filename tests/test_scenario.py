"""Tests for scenario parsing and validation."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from sayagain.scenario import ScenarioError, load_scenario, load_scenarios

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def minimal_dict() -> dict[str, Any]:
    """Smallest scenario that must parse, used as a base for negative cases."""
    return {
        "id": "minimal",
        "description": "A minimal scenario.",
        "agent": {
            "system_prompt": "You are a test agent.",
            "tools": [{"name": "do_thing", "schema": {"date": "string"}}],
        },
        "matrix": {"language": ["en-US"]},
        "turns": [
            {
                "user": {
                    "intent": "do_the_thing",
                    "variants": {"formal": {"en-US": "Please do the thing."}},
                },
                "expect": {"tool_call": {"name": "do_thing", "arguments": {"date": "friday"}}},
            }
        ],
    }


def write(tmp_path: Path, data: dict[str, Any], name: str = "s.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


# --- the headline case: the documented example must load -------------------


def test_loads_the_reschedule_example() -> None:
    scenario = load_scenario(EXAMPLES / "reschedule_appointment.yaml")

    assert scenario.id == "reschedule_appointment"
    assert scenario.repeats == 3
    assert scenario.matrix.language == ["en-US", "en-IN", "es-MX", "pt-BR", "hi-IN", "de-DE"]
    assert [p.id for p in scenario.matrix.perturbation] == [
        "clean",
        "telephone",
        "cafe_10db",
        "street_5db",
        "fast",
        "choppy",
    ]
    assert len(scenario.turns) == 1


def test_tool_schema_is_reachable_despite_the_reserved_name() -> None:
    scenario = load_scenario(EXAMPLES / "reschedule_appointment.yaml")

    tool = scenario.agent.tools[0]
    assert tool.name == "reschedule_appointment"
    assert tool.parameters == {"date": "string", "time": "string"}


def test_interrupt_with_is_reachable_despite_the_reserved_name() -> None:
    scenario = load_scenario(EXAMPLES / "reschedule_appointment.yaml")

    interrupt = scenario.turns[0].interrupt
    assert interrupt is not None
    assert interrupt.after_agent_speaks_ms == 800
    assert interrupt.text["pt-BR"] == "Desculpa — sexta, não quinta."


def test_variants_keep_register_then_language_shape() -> None:
    scenario = load_scenario(EXAMPLES / "reschedule_appointment.yaml")

    variants = scenario.turns[0].user.variants
    assert set(variants) == {"formal", "casual", "disfluent", "codeswitch"}
    assert variants["casual"]["pt-BR"] == "Dá pra jogar minha consulta pra sexta de manhã?"
    assert "pt-BR" not in variants["codeswitch"]


def test_expectations_are_parsed() -> None:
    scenario = load_scenario(EXAMPLES / "reschedule_appointment.yaml")

    expect = scenario.turns[0].expect
    assert expect.tool_call is not None
    assert expect.tool_call.arguments == {"date": "friday", "time": "morning"}
    assert expect.max_first_audio_ms == 1500
    assert expect.max_barge_in_stop_ms == 500
    assert expect.end_state == {"appointment.day": "friday"}


def test_system_prompt_is_read_relative_to_the_scenario_file() -> None:
    scenario = load_scenario(EXAMPLES / "reschedule_appointment.yaml")

    prompt = scenario.system_prompt()
    assert prompt is not None
    assert prompt.startswith("You are the scheduling assistant")


def test_inline_system_prompt_is_returned_as_is(tmp_path: Path) -> None:
    scenario = load_scenario(write(tmp_path, minimal_dict()))

    assert scenario.system_prompt() == "You are a test agent."


# --- ergonomics -----------------------------------------------------------


def test_perturbation_accepts_a_bare_string(tmp_path: Path) -> None:
    data = minimal_dict()
    data["matrix"]["perturbation"] = ["clean", "telephone"]

    scenario = load_scenario(write(tmp_path, data))

    assert [p.id for p in scenario.matrix.perturbation] == ["clean", "telephone"]
    assert scenario.matrix.perturbation[0].params == {}


def test_perturbation_carries_inline_params(tmp_path: Path) -> None:
    data = minimal_dict()
    data["matrix"]["perturbation"] = [{"id": "noise", "params": {"kind": "cafe", "snr_db": 10}}]

    scenario = load_scenario(write(tmp_path, data))

    assert scenario.matrix.perturbation[0].params == {"kind": "cafe", "snr_db": 10}


def test_perturbation_defaults_to_clean(tmp_path: Path) -> None:
    scenario = load_scenario(write(tmp_path, minimal_dict()))

    assert [p.id for p in scenario.matrix.perturbation] == ["clean"]


def test_repeats_defaults_to_one(tmp_path: Path) -> None:
    scenario = load_scenario(write(tmp_path, minimal_dict()))

    assert scenario.repeats == 1


def test_voice_default_stays_a_sentinel(tmp_path: Path) -> None:
    scenario = load_scenario(write(tmp_path, minimal_dict()))

    assert scenario.matrix.voices_for("en-US") is None


def test_a_single_voice_id_is_normalised_to_a_list(tmp_path: Path) -> None:
    data = minimal_dict()
    data["matrix"]["voice"] = {"en-US": "en-US-JennyNeural"}

    scenario = load_scenario(write(tmp_path, data))

    assert scenario.matrix.voices_for("en-US") == ["en-US-JennyNeural"]
    assert scenario.matrix.voices_for("pt-BR") is None


# --- validation: things that would otherwise fail silently ----------------


def test_rejects_a_malformed_language_tag(tmp_path: Path) -> None:
    data = minimal_dict()
    data["matrix"]["language"] = ["en_US"]

    with pytest.raises(ScenarioError, match="en_US"):
        load_scenario(write(tmp_path, data))


def test_rejects_zero_repeats(tmp_path: Path) -> None:
    data = minimal_dict()
    data["repeats"] = 0

    with pytest.raises(ScenarioError, match="repeats"):
        load_scenario(write(tmp_path, data))


def test_rejects_an_expected_tool_the_agent_does_not_declare(tmp_path: Path) -> None:
    data = minimal_dict()
    data["turns"][0]["expect"]["tool_call"]["name"] = "cancel_thing"

    with pytest.raises(ScenarioError, match="cancel_thing"):
        load_scenario(write(tmp_path, data))


def test_rejects_a_turn_whose_variants_cover_no_matrix_language(tmp_path: Path) -> None:
    data = minimal_dict()
    data["matrix"]["language"] = ["de-DE"]

    with pytest.raises(ScenarioError, match="no variant"):
        load_scenario(write(tmp_path, data))


def test_rejects_a_missing_system_prompt_file(tmp_path: Path) -> None:
    data = minimal_dict()
    del data["agent"]["system_prompt"]
    data["agent"]["system_prompt_file"] = "nope.md"

    with pytest.raises(ScenarioError, match=r"nope\.md"):
        load_scenario(write(tmp_path, data))


def test_rejects_declaring_both_prompt_forms(tmp_path: Path) -> None:
    data = minimal_dict()
    data["agent"]["system_prompt_file"] = "prompts/clinic.md"

    with pytest.raises(ScenarioError, match="system_prompt"):
        load_scenario(write(tmp_path, data))


def test_error_names_the_offending_file(tmp_path: Path) -> None:
    data = minimal_dict()
    data["repeats"] = 0
    path = write(tmp_path, data, name="broken.yaml")

    with pytest.raises(ScenarioError, match=r"broken\.yaml"):
        load_scenario(path)


def test_missing_language_in_a_variant_is_not_an_error(tmp_path: Path) -> None:
    data = minimal_dict()
    data["matrix"]["language"] = ["en-US", "de-DE"]

    scenario = load_scenario(write(tmp_path, data))

    assert scenario.matrix.language == ["en-US", "de-DE"]


def test_unused_variant_languages_are_reported_not_rejected(tmp_path: Path) -> None:
    data = minimal_dict()
    data["turns"][0]["user"]["variants"]["formal"]["fr-FR"] = "Faites la chose."

    scenario = load_scenario(write(tmp_path, data))

    assert scenario.unused_variant_languages() == ["fr-FR"]


# --- directory loading ----------------------------------------------------


def test_loads_a_directory_of_scenarios_in_stable_order(tmp_path: Path) -> None:
    write(tmp_path, {**minimal_dict(), "id": "second"}, name="b.yaml")
    write(tmp_path, {**minimal_dict(), "id": "first"}, name="a.yml")

    scenarios = load_scenarios(tmp_path)

    assert [s.id for s in scenarios] == ["first", "second"]


def test_loading_a_single_file_through_load_scenarios(tmp_path: Path) -> None:
    path = write(tmp_path, minimal_dict())

    assert [s.id for s in load_scenarios(path)] == ["minimal"]


def test_empty_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError, match="no scenarios"):
        load_scenarios(tmp_path)
