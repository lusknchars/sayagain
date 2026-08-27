"""Tests for driving one case through an adapter."""

import json
from pathlib import Path

from sayagain.adapters.mock import MockAdapter
from sayagain.expand import Case, expand
from sayagain.runner import RunLog, Runner
from sayagain.scenario import load_scenario
from sayagain.tts import Synthesizer, ToneBackend

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "reschedule_appointment.yaml"
HEARD = "I would like to reschedule my appointment to Friday morning."


class StubTranscriber:
    def __init__(self, text: str = HEARD) -> None:
        self.text = text

    def transcribe(self, pcm: bytes, language: str | None = None) -> str:
        return self.text


class BrokenAdapter:
    name = "broken"

    async def session(self, *, system_prompt: str | None, tools: list[dict[str, object]]) -> object:
        raise RuntimeError("the agent refused the connection")


def make_case(perturbation: str = "clean") -> Case:
    scenario = load_scenario(EXAMPLE)
    return expand(
        scenario,
        languages=["en-US"],
        registers=["formal"],
        perturbations=[perturbation],
        repeats=1,
    )[0]


def make_runner(tmp_path: Path, *, reply_ms: int = 100, **kwargs: object) -> Runner:
    return Runner(
        MockAdapter(transcriber=StubTranscriber(), reply_ms=reply_ms),
        Synthesizer(ToneBackend(), cache_dir=tmp_path / "tts"),
        realtime=False,
        **kwargs,  # type: ignore[arg-type]
    )


async def test_it_records_the_tool_call_the_agent_made(tmp_path: Path) -> None:
    log = await make_runner(tmp_path).run(make_case())

    calls = [event.tool_call for event in log.events if event.kind == "tool_call"]
    assert calls, "the agent never called a tool"
    assert calls[-1] is not None
    assert calls[-1].name == "reschedule_appointment"


async def test_it_records_what_the_user_said(tmp_path: Path) -> None:
    case = make_case()

    log = await make_runner(tmp_path).run(case)

    assert log.turns[0].text == case.turn_texts[0]
    assert log.turns[0].audio_ms > 0
    assert log.turns[0].user_audio_end_ns > 0


async def test_it_records_which_perturbation_ran(tmp_path: Path) -> None:
    log = await make_runner(tmp_path).run(make_case("cafe_10db"))

    assert log.perturbation == {"id": "cafe_10db", "kind": "cafe", "snr_db": 10.0}


async def test_it_captures_the_agent_state(tmp_path: Path) -> None:
    log = await make_runner(tmp_path).run(make_case())

    assert log.state == {"appointment.day": "friday", "appointment.time": "morning"}


async def test_the_agent_speaks_and_ends_its_turn(tmp_path: Path) -> None:
    log = await make_runner(tmp_path).run(make_case())

    kinds = [event.kind for event in log.events]
    assert "audio" in kinds
    assert "end_turn" in kinds
    assert log.error is None


async def test_an_adapter_that_fails_is_recorded_not_raised(tmp_path: Path) -> None:
    runner = Runner(
        BrokenAdapter(),  # type: ignore[arg-type]
        Synthesizer(ToneBackend(), cache_dir=tmp_path / "tts"),
        realtime=False,
    )

    log = await runner.run(make_case())

    assert log.error is not None
    assert "refused the connection" in log.error
    assert log.events == []


async def test_it_writes_a_replayable_log(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, log_dir=tmp_path / "logs")

    log = await runner.run(make_case())

    path = tmp_path / "logs" / f"{log.case_id.replace('/', '_')}.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["type"] == "case"
    assert lines[0]["case_id"] == log.case_id
    assert [line for line in lines if line.get("kind") == "tool_call"]


async def test_it_barges_in_when_the_scenario_asks(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, reply_ms=3000)

    log = await runner.run(make_case())

    turn = log.turns[0]
    assert turn.interrupt_text == "Sorry — Friday, not Thursday."
    assert turn.interrupt_start_ns is not None
    first_audio = next(event.t_ns for event in log.events if event.kind == "audio")
    assert turn.interrupt_start_ns > first_audio


def test_a_run_log_is_json_serialisable() -> None:
    log = RunLog(
        case_id="x/en-US/clean/formal/1",
        scenario_id="x",
        language="en-US",
        register="formal",
        repeat=1,
        perturbation={"id": "clean"},
        seed=1,
    )

    assert json.loads(json.dumps(log.header()))["case_id"] == "x/en-US/clean/formal/1"
