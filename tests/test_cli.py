"""Tests for the command line."""

import os
from pathlib import Path

from typer.testing import CliRunner

from sayagain.cli import app
from sayagain.scenario import load_scenario

runner = CliRunner()
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_run_reports_the_size_of_the_matrix() -> None:
    result = runner.invoke(app, ["run", str(EXAMPLES), "--adapter", "mock", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "reschedule_appointment" in result.output
    assert "126" in result.output


def test_run_applies_the_filters() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(EXAMPLES),
            "--only-language",
            "pt-BR",
            "--only-perturbation",
            "clean",
            "--repeats",
            "1",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "3" in result.output
    assert "hi-IN" not in result.output


def test_run_fails_loudly_on_a_broken_scenario(tmp_path: Path) -> None:
    (tmp_path / "broken.yaml").write_text("id: broken\nnope: true\n", encoding="utf-8")

    result = runner.invoke(app, ["run", str(tmp_path), "--dry-run"])

    assert result.exit_code != 0
    assert "broken.yaml" in result.output


def test_run_rejects_an_unknown_adapter() -> None:
    result = runner.invoke(app, ["run", str(EXAMPLES), "--adapter", "nope", "--dry-run"])

    assert result.exit_code != 0
    assert "nope" in result.output


def test_new_writes_a_scenario_that_loads(tmp_path: Path) -> None:
    result = runner.invoke(app, ["new", "order_status", "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    scenario = load_scenario(tmp_path / "order_status.yaml")
    assert scenario.id == "order_status"


def test_new_refuses_to_overwrite(tmp_path: Path) -> None:
    runner.invoke(app, ["new", "order_status", "--out", str(tmp_path)])

    result = runner.invoke(app, ["new", "order_status", "--out", str(tmp_path)])

    assert result.exit_code != 0
    assert "exists" in result.output


def test_doctor_reports_the_environment() -> None:
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "ffmpeg" in result.output


def test_run_rejects_an_unknown_tts_backend() -> None:
    result = runner.invoke(app, ["run", str(EXAMPLES), "--tts", "elevenlabs"])

    assert result.exit_code != 0
    assert "elevenlabs" in result.output


def test_voices_lists_the_offline_backend() -> None:
    result = runner.invoke(app, ["voices", "--backend", "tone"])

    assert result.exit_code == 0, result.output
    assert "tone" in result.output


def test_voices_rejects_an_unknown_backend() -> None:
    result = runner.invoke(app, ["voices", "--backend", "nope"])

    assert result.exit_code != 0
    assert "nope" in result.output


def test_websocket_without_a_url_fails_before_it_connects() -> None:
    result = runner.invoke(app, ["run", str(EXAMPLES), "--adapter", "websocket"])

    assert result.exit_code != 0
    assert "url" in result.output.lower()


def test_every_documented_adapter_is_selectable() -> None:
    result = runner.invoke(app, ["run", str(EXAMPLES), "--adapter", "openai_realtime", "--dry-run"])

    assert result.exit_code == 0, result.output


def test_openai_realtime_without_a_key_fails_before_running_anything() -> None:
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        result = runner.invoke(app, ["run", str(EXAMPLES), "--adapter", "openai_realtime"])
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved

    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in result.output


def test_run_can_filter_to_one_register() -> None:
    result = runner.invoke(
        app,
        ["run", str(EXAMPLES), "--only-register", "disfluent", "--repeats", "1", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "12" in result.output  # 2 languages x 6 perturbations
