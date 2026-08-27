"""Tests for metrics, failure attribution, and aggregation."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sayagain.adapters.base import AgentEvent, ToolCall
from sayagain.expand import Case, expand
from sayagain.runner import RunLog, TurnRecord
from sayagain.scenario import load_scenario
from sayagain.score import CaseResult, aggregate, score

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "reschedule_appointment.yaml"
MS = 1_000_000
SPOKEN = "Gostaria de reagendar minha consulta para sexta de manhã."


def make_case(language: str = "pt-BR", register: str = "formal", repeat: int = 1) -> Case:
    scenario = load_scenario(EXAMPLE)
    cases = expand(
        scenario, languages=[language], registers=[register], perturbations=["clean"], repeats=3
    )
    return next(case for case in cases if case.repeat == repeat)


def make_log(
    case: Case,
    *,
    transcript: str | None = SPOKEN,
    arguments: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    first_audio_ms: int = 400,
    interrupt_ms: int | None = None,
    stop_after_ms: int = 200,
    error: str | None = None,
) -> RunLog:
    user_end = 1000 * MS
    events: list[AgentEvent] = []
    if transcript is not None:
        events.append(AgentEvent(kind="transcript", t_ns=user_end + 10 * MS, text=transcript))
    if arguments is not None:
        call = ToolCall(
            name="reschedule_appointment", arguments=dict(arguments), t_ns=user_end + 20 * MS
        )
        events.append(AgentEvent(kind="tool_call", t_ns=call.t_ns, tool_call=call))
    speech_start = user_end + first_audio_ms * MS
    for index in range(5):
        events.append(
            AgentEvent(kind="audio", t_ns=speech_start + index * 20 * MS, audio=b"\x10\x10" * 320)
        )
    interrupt_ns = None
    if interrupt_ms is not None:
        interrupt_ns = speech_start + interrupt_ms * MS
        for index in range(3):
            events.append(
                AgentEvent(
                    kind="audio",
                    t_ns=interrupt_ns + index * (stop_after_ms // 3) * MS,
                    audio=b"\x10\x10" * 320,
                )
            )
    events.append(AgentEvent(kind="end_turn", t_ns=speech_start + 5_000 * MS))
    events.sort(key=lambda event: event.t_ns)
    return RunLog(
        case_id=case.id,
        scenario_id=case.scenario.id,
        language=case.language,
        register=case.register,
        repeat=case.repeat,
        perturbation={"id": "clean"},
        seed=1,
        turns=[
            TurnRecord(
                text=case.turn_texts[0],
                audio_ms=3000,
                user_audio_end_ns=user_end,
                interrupt_text=case.interrupt_texts[0] if interrupt_ms is not None else None,
                interrupt_start_ns=interrupt_ns,
            )
        ],
        events=events,
        state=dict(state or {}),
        error=error,
    )


CORRECT_ARGS = {"date": "sexta-feira", "time": "de manhã"}
CORRECT_STATE = {"appointment.day": "sexta", "appointment.time": "manhã"}


# --- the happy path -------------------------------------------------------


def test_a_correct_run_passes() -> None:
    case = make_case()

    result = score(make_log(case, arguments=CORRECT_ARGS, state=CORRECT_STATE), case)

    assert result.passed is True
    assert result.tool_call_correct is True
    assert result.end_state_correct is True
    assert result.failure_locus == "none"


def test_arguments_are_matched_through_the_normalizer() -> None:
    case = make_case()

    result = score(make_log(case, arguments={"date": "SEXTA!", "time": "manha"}), case)

    assert result.tool_call_correct is True


# --- failure attribution --------------------------------------------------


def test_a_good_transcript_with_a_bad_tool_call_is_a_reasoning_failure() -> None:
    case = make_case()

    result = score(make_log(case, arguments={"date": "quinta", "time": "manhã"}), case)

    assert result.tool_call_correct is False
    assert result.failure_locus == "reasoning"
    assert result.passed is False


def test_a_garbled_transcript_with_a_bad_tool_call_is_an_asr_failure() -> None:
    case = make_case()

    result = score(
        make_log(case, transcript="cachorro pizza telefone azul", arguments={"date": "quinta"}),
        case,
    )

    assert result.failure_locus == "asr"
    assert result.transcript_wer is not None
    assert result.transcript_wer > 0.5


def test_no_transcript_means_the_locus_is_unknown() -> None:
    case = make_case()

    result = score(make_log(case, transcript=None, arguments={"date": "quinta"}), case)

    assert result.failure_locus == "unknown"
    assert result.transcript_wer is None


def test_no_tool_call_at_all_still_fails() -> None:
    case = make_case()

    result = score(make_log(case, arguments=None), case)

    assert result.tool_call_correct is False
    assert result.passed is False


def test_the_last_tool_call_wins_because_corrections_come_last() -> None:
    case = make_case()
    log = make_log(case, arguments={"date": "quinta", "time": "manhã"})
    late = ToolCall(name="reschedule_appointment", arguments=CORRECT_ARGS, t_ns=9_000 * MS)
    log.events.append(AgentEvent(kind="tool_call", t_ns=late.t_ns, tool_call=late))

    result = score(log, case)

    assert result.tool_call_correct is True


def test_an_error_fails_the_case_without_pretending_to_measure() -> None:
    case = make_case()

    result = score(make_log(case, arguments=CORRECT_ARGS, error="ConnectionError: refused"), case)

    assert result.passed is False
    assert any(not assertion.passed and assertion.name == "ran" for assertion in result.assertions)


# --- latency --------------------------------------------------------------


def test_first_audio_is_measured_from_the_end_of_the_user_utterance() -> None:
    case = make_case()

    result = score(make_log(case, arguments=CORRECT_ARGS, first_audio_ms=400), case)

    assert result.first_audio_ms is not None
    assert abs(result.first_audio_ms - 400) < 1


def test_a_slow_first_response_fails_the_latency_expectation() -> None:
    case = make_case()  # the example allows 1500 ms

    result = score(make_log(case, arguments=CORRECT_ARGS, first_audio_ms=2500), case)

    assert result.passed is False
    assert any(
        assertion.name == "max_first_audio_ms" and not assertion.passed
        for assertion in result.assertions
    )


def test_silence_from_the_agent_is_a_latency_failure_not_a_crash() -> None:
    case = make_case()
    log = make_log(case, arguments=CORRECT_ARGS)
    log.events = [event for event in log.events if event.kind != "audio"]

    result = score(log, case)

    assert result.first_audio_ms is None
    assert result.passed is False


# --- barge-in -------------------------------------------------------------


def test_barge_in_stop_is_measured_from_the_interruption() -> None:
    case = make_case()

    result = score(
        make_log(case, arguments=CORRECT_ARGS, interrupt_ms=100, stop_after_ms=300), case
    )

    assert result.barge_in_stop_ms is not None
    assert 100 < result.barge_in_stop_ms < 400


def test_an_agent_that_was_not_speaking_cannot_talk_over_you() -> None:
    case = make_case()

    result = score(make_log(case, arguments=CORRECT_ARGS, interrupt_ms=None), case)

    assert result.barge_in_stop_ms is None
    assert all(
        assertion.passed
        for assertion in result.assertions
        if assertion.name == "max_barge_in_stop_ms"
    )


# --- aggregation ----------------------------------------------------------


def build_results(passing: int, failing: int) -> list[CaseResult]:
    results: list[CaseResult] = []
    for index in range(passing + failing):
        case = make_case(repeat=(index % 3) + 1)
        arguments = CORRECT_ARGS if index < passing else {"date": "quinta"}
        results.append(score(make_log(case, arguments=arguments, state=CORRECT_STATE), case))
    return results


def test_aggregate_reports_a_pass_rate_per_language_and_perturbation() -> None:
    rows = aggregate(build_results(2, 1))

    assert len(rows) == 1
    row = rows[0]
    assert row.labels == {"language": "pt-BR", "perturbation": "clean"}
    assert row.cases == 3
    assert abs(row.pass_rate - 2 / 3) < 1e-9


def test_aggregate_can_group_by_register_because_that_is_where_the_signal_is() -> None:
    rows = aggregate(build_results(2, 1), by=("language", "register"))

    assert len(rows) == 1
    assert rows[0].labels == {"language": "pt-BR", "register": "formal"}


def test_aggregate_reports_a_rate_per_assertion_not_just_overall() -> None:
    """One failing assertion must not be allowed to hide the other three."""
    rows = aggregate(build_results(2, 1))

    rates = rows[0].assertion_rates
    assert rates["tool_call"] == 2 / 3
    assert rates["max_first_audio_ms"] == 1.0  # latency was fine in every case
    assert set(rates) == {"tool_call", "end_state", "max_first_audio_ms", "max_barge_in_stop_ms"}


def test_an_assertion_a_scenario_never_makes_is_absent_not_zero() -> None:
    case = make_case()
    log = make_log(case, arguments=CORRECT_ARGS, state=CORRECT_STATE)
    log.turns[0] = TurnRecord(
        text=log.turns[0].text,
        audio_ms=log.turns[0].audio_ms,
        user_audio_end_ns=log.turns[0].user_audio_end_ns,
    )

    rows = aggregate([score(log, case)])

    assert "max_barge_in_stop_ms" in rows[0].assertion_rates  # asserted, just not applicable


def test_pass_at_k_is_lenient_and_pass_hat_k_is_strict() -> None:
    rows = aggregate(build_results(2, 1))

    assert rows[0].pass_at_k == 1.0  # at least one of the three repeats passed
    assert rows[0].pass_hat_k == 0.0  # but not all of them did


def test_aggregate_reports_latency_percentiles_and_loci() -> None:
    rows = aggregate(build_results(3, 0))

    assert rows[0].p50_first_audio_ms is not None
    assert rows[0].p95_first_audio_ms is not None
    assert rows[0].locus_counts == {"none": 3}


def test_aggregate_of_nothing_is_nothing() -> None:
    assert aggregate([]) == []


def test_an_interruption_after_the_agent_finished_is_not_a_barge_in() -> None:
    """Answering a correction is not the same as talking over the person making it.

    The agent speaks, ends its turn, and only then is interrupted. The audio it
    emits afterwards is its reply to the correction, and scoring that as
    "failed to stop" makes every well-behaved agent look broken.
    """
    case = make_case()
    log = make_log(case, arguments=CORRECT_ARGS, state=CORRECT_STATE)
    speech_end = max(event.t_ns for event in log.events if event.kind == "audio")
    log.events = [event for event in log.events if event.kind != "end_turn"]
    log.events.append(AgentEvent(kind="end_turn", t_ns=speech_end + 20 * MS))

    interrupt_ns = speech_end + 500 * MS
    log.turns[0] = TurnRecord(
        text=log.turns[0].text,
        audio_ms=log.turns[0].audio_ms,
        user_audio_end_ns=log.turns[0].user_audio_end_ns,
        interrupt_text="Sorry — Friday, not Thursday.",
        interrupt_start_ns=interrupt_ns,
    )
    for index in range(5):  # the reply to the correction
        log.events.append(
            AgentEvent(
                kind="audio", t_ns=interrupt_ns + (600 + index * 20) * MS, audio=b"\x10\x10" * 320
            )
        )
    log.events.append(AgentEvent(kind="end_turn", t_ns=interrupt_ns + 900 * MS))
    log.events.sort(key=lambda event: event.t_ns)

    result = score(log, case)

    assert result.barge_in_stop_ms is None
    assert result.passed is True
