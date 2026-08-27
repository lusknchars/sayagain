"""Tests for the report artifacts."""

import json
from pathlib import Path

from sayagain.report import write_report
from sayagain.score import Assertion, CaseResult

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def result(
    *,
    register: str = "formal",
    language: str = "en-US",
    perturbation: str = "clean",
    repeat: int = 1,
    passed: bool = True,
    locus: str = "none",
) -> CaseResult:
    assertions = [
        Assertion("tool_call", passed, "wanted reschedule_appointment{'date': 'friday'}"),
        Assertion("max_first_audio_ms", True, "wanted <= 1500 ms, got 400 ms"),
    ]
    return CaseResult(
        case_id=f"demo/{language}/{perturbation}/{register}/{repeat}",
        scenario_id="demo",
        language=language,
        register=register,
        perturbation=perturbation,
        repeat=repeat,
        failure_locus=locus,  # type: ignore[arg-type]
        assertions=assertions,
        first_audio_ms=400.0,
        tool_call_correct=passed,
        transcript_wer=0.0 if passed else 0.8,
        transcript="Um, so, I was, like, wondering if I could move it?",
        error=None,
    )


def mixed() -> list[CaseResult]:
    return [
        result(register="formal"),
        result(register="casual"),
        result(register="disfluent", passed=False, locus="asr"),
        result(register="disfluent", perturbation="cafe_10db", passed=False, locus="asr"),
        result(register="formal", perturbation="cafe_10db"),
        result(language="pt-BR", register="formal"),
    ]


def test_it_writes_the_three_artifacts(tmp_path: Path) -> None:
    paths = write_report(mixed(), out_dir=tmp_path, scenario_id="demo")

    assert paths.json.is_file()
    assert paths.markdown.is_file()
    assert paths.heatmap is not None
    assert paths.heatmap.read_bytes()[:8] == PNG_MAGIC


def test_the_json_holds_every_case(tmp_path: Path) -> None:
    paths = write_report(mixed(), out_dir=tmp_path, scenario_id="demo")

    payload = json.loads(paths.json.read_text())
    assert payload["scenario_id"] == "demo"
    assert payload["totals"]["cases"] == 6
    assert payload["totals"]["failed"] == 2
    assert len(payload["cases"]) == 6
    assert payload["cases"][0]["assertions"][0]["name"] == "tool_call"


def test_the_json_reports_rates_per_assertion(tmp_path: Path) -> None:
    paths = write_report(mixed(), out_dir=tmp_path, scenario_id="demo")

    rows = json.loads(paths.json.read_text())["by_language_and_perturbation"]
    clean = next(row for row in rows if row["labels"]["perturbation"] == "clean")
    assert clean["assertion_rates"]["max_first_audio_ms"] == 1.0
    assert clean["assertion_rates"]["tool_call"] < 1.0


def test_the_markdown_leads_with_what_broke(tmp_path: Path) -> None:
    paths = write_report(mixed(), out_dir=tmp_path, scenario_id="demo")

    text = paths.markdown.read_text()
    assert text.index("## What broke") < text.index("## Pass rate by assertion")
    assert "Um, so, I was, like, wondering if I could move it?" in text
    assert "disfluent" in text


def test_the_markdown_reports_registers_separately(tmp_path: Path) -> None:
    paths = write_report(mixed(), out_dir=tmp_path, scenario_id="demo")

    assert "## Pass rate by register" in paths.markdown.read_text()


def test_a_clean_run_says_so_without_a_failure_section(tmp_path: Path) -> None:
    paths = write_report(
        [result(), result(register="casual")], out_dir=tmp_path, scenario_id="demo"
    )

    text = paths.markdown.read_text()
    assert "## What broke" not in text
    assert "every case passed" in text.lower()


def test_an_empty_run_does_not_crash(tmp_path: Path) -> None:
    paths = write_report([], out_dir=tmp_path, scenario_id="demo")

    assert paths.json.is_file()
    assert json.loads(paths.json.read_text())["totals"]["cases"] == 0


def test_the_report_records_how_it_was_run(tmp_path: Path) -> None:
    paths = write_report(
        mixed(), out_dir=tmp_path, scenario_id="demo", settings={"adapter": "mock", "seed": 42}
    )

    payload = json.loads(paths.json.read_text())
    assert payload["settings"]["adapter"] == "mock"
    assert "mock" in paths.markdown.read_text()
