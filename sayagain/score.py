"""Turn a run log into metrics, and say where the agent broke.

The single most useful thing this file produces is `failure_locus`. Knowing a
case failed is worth little; knowing whether the words never arrived (`asr`) or
arrived and were misused (`reasoning`) is what tells you which half of your
stack to go and fix.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np

from sayagain.adapters.base import ToolCall
from sayagain.audio import has_speech
from sayagain.expand import Case
from sayagain.normalize import arguments_match, normalise_value, tokens
from sayagain.runner import RunLog, TurnRecord
from sayagain.scenario import Expect

FailureLocus = Literal["none", "asr", "reasoning", "unknown"]

#: A transcript with a word error rate above this counts as "the words did not
#: arrive". It is a judgement call, so `transcript_wer` is always reported too.
ASR_ERROR_THRESHOLD = 0.2

NANOS_PER_MS = 1_000_000


@dataclass(frozen=True, slots=True)
class Assertion:
    """One thing the scenario asked for, and whether it happened."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Everything measurable about one run of one case."""

    case_id: str
    scenario_id: str
    language: str
    register: str
    perturbation: str
    repeat: int
    failure_locus: FailureLocus
    assertions: list[Assertion] = field(default_factory=list)
    first_audio_ms: float | None = None
    barge_in_stop_ms: float | None = None
    tool_call_correct: bool | None = None
    end_state_correct: bool | None = None
    transcript_wer: float | None = None
    transcript: str | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        """True when every assertion the scenario made held."""
        return all(assertion.passed for assertion in self.assertions)

    @property
    def group(self) -> tuple[str, str, str]:
        """The repeat group this case belongs to: everything but the repeat number."""
        return (self.language, self.perturbation, self.register)


def score(log: RunLog, case: Case, *, today: date | None = None) -> CaseResult:
    """Measure one run against what its scenario expected."""
    expect = case.scenario.turns[0].expect
    turn = log.turns[0] if log.turns else None

    transcript = _heard(log)
    spoken = turn.text if turn else case.turn_texts[0]
    wer = word_error_rate(spoken, transcript) if transcript is not None else None

    call = _last_tool_call(log)
    tool_ok = _tool_call_correct(expect, call, case, today) if expect.tool_call else None
    state_ok = (
        _end_state_correct(expect.end_state, log.state, case, today) if expect.end_state else None
    )

    first_audio = _first_audio_ms(log, turn)
    barge_stop = _barge_in_stop_ms(log, turn)

    assertions: list[Assertion] = []
    if log.error is not None:
        assertions.append(Assertion("ran", False, log.error))
    if expect.tool_call is not None:
        assertions.append(
            Assertion(
                "tool_call",
                bool(tool_ok),
                _describe_call(call, expect.tool_call.name, expect.tool_call.arguments),
            )
        )
    if expect.end_state:
        assertions.append(
            Assertion("end_state", bool(state_ok), f"wanted {expect.end_state}, got {log.state}")
        )
    if expect.max_first_audio_ms is not None:
        ok = first_audio is not None and first_audio <= expect.max_first_audio_ms
        got = f"{first_audio:.0f} ms" if first_audio is not None else "the agent never spoke"
        assertions.append(
            Assertion(
                "max_first_audio_ms",
                ok,
                f"wanted <= {expect.max_first_audio_ms} ms, got {got}",
            )
        )
    if expect.max_barge_in_stop_ms is not None:
        if barge_stop is None:
            assertions.append(
                Assertion(
                    "max_barge_in_stop_ms",
                    True,
                    "the agent was not speaking, nothing to stop",
                )
            )
        else:
            assertions.append(
                Assertion(
                    "max_barge_in_stop_ms",
                    barge_stop <= expect.max_barge_in_stop_ms,
                    f"wanted <= {expect.max_barge_in_stop_ms} ms, got {barge_stop:.0f} ms",
                )
            )

    return CaseResult(
        case_id=log.case_id,
        scenario_id=log.scenario_id,
        language=log.language,
        register=log.register,
        perturbation=str(log.perturbation.get("id", "unknown")),
        repeat=log.repeat,
        failure_locus=_locus(tool_ok, state_ok, transcript, wer),
        assertions=assertions,
        first_audio_ms=first_audio,
        barge_in_stop_ms=barge_stop,
        tool_call_correct=tool_ok,
        end_state_correct=state_ok,
        transcript_wer=wer,
        transcript=transcript,
        error=log.error,
    )


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, divided by the reference length."""
    want = tokens(reference)
    got = tokens(hypothesis)
    if not want:
        return 0.0 if not got else 1.0

    previous = list(range(len(got) + 1))
    for i, want_word in enumerate(want, start=1):
        current = [i]
        for j, got_word in enumerate(got, start=1):
            cost = 0 if want_word == got_word else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return float(previous[-1]) / len(want)


@dataclass(frozen=True, slots=True)
class Aggregate:
    """One row of the report: everything that ran under the same conditions."""

    labels: dict[str, str]
    cases: int
    pass_rate: float
    pass_at_k: float
    pass_hat_k: float
    assertion_rates: dict[str, float]
    locus_counts: dict[str, int]
    p50_first_audio_ms: float | None = None
    p95_first_audio_ms: float | None = None
    p50_barge_in_stop_ms: float | None = None
    p95_barge_in_stop_ms: float | None = None

    @property
    def top_failure_locus(self) -> str:
        """The most common reason cases in this row failed, or `none`."""
        failures = {k: v for k, v in self.locus_counts.items() if k != "none"}
        if not failures:
            return "none"
        return max(failures.items(), key=lambda item: item[1])[0]


def aggregate(
    results: list[CaseResult], *, by: tuple[str, ...] = ("language", "perturbation")
) -> list[Aggregate]:
    """Group results along the given axes.

    The default groups the way the spec's table does. Grouping by `register`
    instead is what surfaces the failures that acoustic conditions do not
    explain, so both are first-class.
    """
    rows: dict[tuple[str, ...], list[CaseResult]] = {}
    for result in results:
        key = tuple(str(getattr(result, axis)) for axis in by)
        rows.setdefault(key, []).append(result)

    out: list[Aggregate] = []
    for key, group in rows.items():
        repeats: dict[tuple[str, str, str], list[bool]] = {}
        for result in group:
            repeats.setdefault(result.group, []).append(result.passed)
        out.append(
            Aggregate(
                labels=dict(zip(by, key, strict=True)),
                cases=len(group),
                pass_rate=sum(result.passed for result in group) / len(group),
                pass_at_k=_fraction(repeats.values(), any),
                pass_hat_k=_fraction(repeats.values(), all),
                assertion_rates=_assertion_rates(group),
                locus_counts=dict(Counter(result.failure_locus for result in group)),
                p50_first_audio_ms=_percentile(group, "first_audio_ms", 50),
                p95_first_audio_ms=_percentile(group, "first_audio_ms", 95),
                p50_barge_in_stop_ms=_percentile(group, "barge_in_stop_ms", 50),
                p95_barge_in_stop_ms=_percentile(group, "barge_in_stop_ms", 95),
            )
        )
    return out


def _assertion_rates(results: list[CaseResult]) -> dict[str, float]:
    """Pass rate per assertion, counting only cases where it was asserted.

    Reporting these separately is the difference between "0%" and "your tool
    calls are fine, your latency budget is wrong".
    """
    passed: Counter[str] = Counter()
    asserted: Counter[str] = Counter()
    for result in results:
        for assertion in result.assertions:
            asserted[assertion.name] += 1
            passed[assertion.name] += assertion.passed
    return {name: passed[name] / count for name, count in asserted.items()}


def _heard(log: RunLog) -> str | None:
    """Return what the agent said it heard, before any interruption."""
    cutoff = log.turns[0].interrupt_start_ns if log.turns else None
    parts = [
        event.text
        for event in log.events
        if event.kind == "transcript" and event.text and (cutoff is None or event.t_ns < cutoff)
    ]
    return " ".join(parts) if parts else None


def _last_tool_call(log: RunLog) -> ToolCall | None:
    calls = [event.tool_call for event in log.events if event.kind == "tool_call"]
    return calls[-1] if calls and calls[-1] is not None else None


def _tool_call_correct(
    expect: Expect, call: ToolCall | None, case: Case, today: date | None
) -> bool:
    if call is None or expect.tool_call is None:
        return False
    if call.name != expect.tool_call.name:
        return False
    return arguments_match(
        expect.tool_call.arguments, call.arguments, language=case.language, today=today
    )


def _end_state_correct(
    wanted: dict[str, Any], got: dict[str, Any], case: Case, today: date | None
) -> bool:
    for path, value in wanted.items():
        if path not in got:
            return False
        field_name = path.rsplit(".", 1)[-1]
        left = normalise_value(value, field=field_name, language=case.language, today=today)
        right = normalise_value(got[path], field=field_name, language=case.language, today=today)
        if left != right:
            return False
    return True


def _first_audio_ms(log: RunLog, turn: TurnRecord | None) -> float | None:
    if turn is None:
        return None
    for event in sorted(log.events, key=lambda item: item.t_ns):
        if event.kind != "audio" or event.t_ns < turn.user_audio_end_ns:
            continue
        if event.audio is not None and not has_speech(event.audio):
            continue
        return (event.t_ns - turn.user_audio_end_ns) / NANOS_PER_MS
    return None


def _barge_in_stop_ms(log: RunLog, turn: TurnRecord | None) -> float | None:
    """How long the agent kept talking after being talked over.

    None means there was nothing to stop: either no interruption was performed,
    or the agent had already ended its turn when the interruption landed. Audio
    it emits after that point is its answer to the correction, and counting that
    as "failed to stop" would fail every agent that behaves correctly.
    """
    if turn is None or turn.interrupt_start_ns is None:
        return None
    start = turn.interrupt_start_ns

    last_audio_before = _last(log, "audio", before=start)
    last_end_before = _last(log, "end_turn", before=start)
    was_speaking = last_audio_before is not None and (
        last_end_before is None or last_audio_before > last_end_before
    )
    if not was_speaking:
        return None

    ends = [event.t_ns for event in log.events if event.kind == "end_turn" and event.t_ns >= start]
    limit = min(ends) if ends else float("inf")
    after = [
        event.t_ns for event in log.events if event.kind == "audio" and start <= event.t_ns <= limit
    ]
    if not after:
        return 0.0  # it was speaking and stopped inside the same frame
    return (max(after) - start) / NANOS_PER_MS


def _last(log: RunLog, kind: str, *, before: int) -> int | None:
    stamps = [event.t_ns for event in log.events if event.kind == kind and event.t_ns < before]
    return max(stamps) if stamps else None


def _locus(
    tool_ok: bool | None, state_ok: bool | None, transcript: str | None, wer: float | None
) -> FailureLocus:
    checked = [value for value in (tool_ok, state_ok) if value is not None]
    if checked and all(checked):
        return "none"
    if not checked:
        return "none"
    if transcript is None or wer is None:
        return "unknown"
    return "asr" if wer > ASR_ERROR_THRESHOLD else "reasoning"


def _describe_call(call: ToolCall | None, name: str, arguments: dict[str, Any]) -> str:
    if call is None:
        return f"wanted {name}{arguments}, the agent called nothing"
    return f"wanted {name}{arguments}, got {call.name}{call.arguments}"


def _fraction(groups: Any, predicate: Any) -> float:
    groups = list(groups)
    if not groups:
        return 0.0
    return float(sum(1 for outcomes in groups if predicate(outcomes))) / len(groups)


def _percentile(results: list[CaseResult], attribute: str, percentile: int) -> float | None:
    values = [
        getattr(result, attribute) for result in results if getattr(result, attribute) is not None
    ]
    if not values:
        return None
    return float(np.percentile(values, percentile))
