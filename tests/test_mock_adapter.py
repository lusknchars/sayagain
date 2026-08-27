"""Tests for the in-process mock agent."""

import asyncio
import time

from sayagain.adapters.base import Adapter, AgentEvent
from sayagain.adapters.mock import MockAdapter
from sayagain.audio import FRAME_BYTES, frames, tone

TOOLS = [{"name": "reschedule_appointment", "schema": {"date": "string", "time": "string"}}]


class StubTranscriber:
    """Stands in for whisper so unit tests do not depend on a model."""

    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, pcm: bytes, language: str | None = None) -> str:
        return self.text


async def run_turn(text: str, *, silence_ms: int = 600) -> list[AgentEvent]:
    adapter = MockAdapter(transcriber=StubTranscriber(text), reply_ms=100)
    session = await adapter.session(system_prompt=None, tools=TOOLS)
    for frame in frames(tone(500)):
        await session.send_audio(frame)
    await session.send_silence(silence_ms)
    await session.close()
    return [event async for event in session.events()]


def kinds(events: list[AgentEvent]) -> list[str]:
    return [event.kind for event in events]


def test_the_mock_is_an_adapter() -> None:
    assert isinstance(MockAdapter(), Adapter)
    assert MockAdapter().name == "mock"


async def test_it_calls_the_tool_the_transcript_asks_for() -> None:
    events = await run_turn("I'd like to reschedule my appointment to Friday morning.")

    calls = [event.tool_call for event in events if event.kind == "tool_call"]
    assert len(calls) == 1
    assert calls[0] is not None
    assert calls[0].name == "reschedule_appointment"
    assert calls[0].arguments == {"date": "friday", "time": "morning"}


async def test_it_reports_what_it_heard() -> None:
    events = await run_turn("I'd like to reschedule my appointment to Friday morning.")

    transcripts = [event.text for event in events if event.kind == "transcript"]
    assert transcripts == ["I'd like to reschedule my appointment to Friday morning."]


async def test_it_understands_portuguese() -> None:
    events = await run_turn("Gostaria de reagendar minha consulta para sexta de manhã.")

    calls = [event.tool_call for event in events if event.kind == "tool_call"]
    assert calls[0] is not None
    assert calls[0].arguments == {"date": "friday", "time": "morning"}


async def test_it_understands_a_code_switched_sentence() -> None:
    events = await run_turn("Can I move it to sexta, like, in the morning?")

    calls = [event.tool_call for event in events if event.kind == "tool_call"]
    assert calls[0] is not None
    assert calls[0].arguments == {"date": "friday", "time": "morning"}


async def test_it_speaks_before_it_ends_the_turn() -> None:
    events = await run_turn("I'd like to reschedule my appointment to Friday morning.")

    assert kinds(events)[-1] == "end_turn"
    audio = [event for event in events if event.kind == "audio"]
    assert audio, "the mock must emit audio so first_audio_ms is measurable"
    assert {len(event.audio or b"") for event in audio} == {FRAME_BYTES}
    assert kinds(events).index("audio") < kinds(events).index("end_turn")


async def test_timestamps_never_go_backwards() -> None:
    events = await run_turn("I'd like to reschedule my appointment to Friday morning.")

    stamps = [event.t_ns for event in events]
    assert stamps == sorted(stamps)


async def test_a_transcript_it_cannot_parse_calls_no_tool() -> None:
    events = await run_turn("Mmm hello is this the, uh, the place?")

    assert "tool_call" not in kinds(events)
    assert kinds(events)[-1] == "end_turn"


async def test_it_waits_for_the_user_to_stop_talking() -> None:
    events = await run_turn("I'd like to reschedule to Friday morning.", silence_ms=100)

    assert events == []


async def test_state_reports_the_effect_of_the_tool_call() -> None:
    adapter = MockAdapter(
        transcriber=StubTranscriber("I'd like to reschedule my appointment to Friday morning.")
    )
    session = await adapter.session(system_prompt=None, tools=TOOLS)
    for frame in frames(tone(500)):
        await session.send_audio(frame)
    await session.send_silence(600)
    await session.close()  # the reply is asynchronous, so wait for it to land

    assert await session.state() == {"appointment.day": "friday", "appointment.time": "morning"}


async def test_state_is_empty_before_anything_happens() -> None:
    session = await MockAdapter().session(system_prompt=None, tools=TOOLS)

    assert await session.state() == {}


async def test_it_stops_speaking_when_the_user_interrupts() -> None:
    adapter = MockAdapter(
        transcriber=StubTranscriber("reschedule my appointment to Friday morning"),
        reply_ms=3000,
    )
    session = await adapter.session(system_prompt=None, tools=TOOLS)
    for frame in frames(tone(200)):
        await session.send_audio(frame)
    await session.send_silence(600)

    collected: list[AgentEvent] = []

    async def drain() -> None:
        async for event in session.events():
            collected.append(event)

    task = asyncio.create_task(drain())
    await asyncio.sleep(0.25)
    spoken_before = sum(1 for event in collected if event.kind == "audio")

    for frame in frames(tone(60)):
        await session.send_audio(frame)
    await asyncio.sleep(0.15)
    spoken_after = sum(1 for event in collected if event.kind == "audio")

    await session.close()
    await task

    # A 3 s reply is 150 frames. If it had queued them all up front, both of
    # these would pass trivially, so both bounds matter.
    assert 0 < spoken_before < 40, "it should still have been mid-reply at 250 ms"
    assert spoken_after - spoken_before <= 3, "it kept talking over the user"
    assert sum(1 for event in collected if event.kind == "audio") < 40
    assert collected[-1].kind == "end_turn"


async def test_it_finishes_speaking_when_nobody_interrupts() -> None:
    adapter = MockAdapter(
        transcriber=StubTranscriber("reschedule my appointment to Friday morning"),
        reply_ms=100,
    )
    session = await adapter.session(system_prompt=None, tools=TOOLS)
    for frame in frames(tone(200)):
        await session.send_audio(frame)
    await session.send_silence(600)
    await session.close()

    events = [event async for event in session.events()]

    assert sum(1 for event in events if event.kind == "audio") == 5


async def test_it_answers_a_follow_up_after_it_has_already_spoken() -> None:
    adapter = MockAdapter(
        transcriber=StubTranscriber("move my appointment to Friday morning"), reply_ms=60
    )
    session = await adapter.session(system_prompt=None, tools=TOOLS)
    for _ in range(2):
        for frame in frames(tone(200)):
            await session.send_audio(frame)
        await session.send_silence(600)
        await asyncio.sleep(0.15)  # let the 60 ms reply actually play out
    await session.close()

    events = [event async for event in session.events()]

    assert kinds(events).count("end_turn") == 2
    assert kinds(events).count("tool_call") == 2
    assert kinds(events).count("audio") == 6  # neither reply was cut off


async def test_transcribing_does_not_block_the_caller() -> None:
    """A real agent is a separate process; the mock must not stall the runner.

    If transcription runs inline, the runner's `send_audio` loop stops while it
    happens, and every timestamp after that point is wrong.
    """

    class SlowTranscriber:
        def transcribe(self, pcm: bytes, language: str | None = None) -> str:
            time.sleep(0.4)
            return "move my appointment to Friday morning"

    adapter = MockAdapter(transcriber=SlowTranscriber(), reply_ms=60)
    session = await adapter.session(system_prompt=None, tools=TOOLS)
    for frame in frames(tone(200)):
        await session.send_audio(frame)

    started = time.perf_counter()
    await session.send_silence(600)
    elapsed = time.perf_counter() - started

    await session.close()
    events = [event async for event in session.events()]

    assert elapsed < 0.2, f"send_silence waited {elapsed:.2f}s for transcription"
    assert any(event.kind == "tool_call" for event in events)


BANK_TOOLS = [{"name": "transfer_money", "schema": {"amount": "string", "date": "string"}}]


async def bank_turn(text: str) -> list[AgentEvent]:
    adapter = MockAdapter(transcriber=StubTranscriber(text), reply_ms=60)
    session = await adapter.session(system_prompt=None, tools=BANK_TOOLS)
    for frame in frames(tone(200)):
        await session.send_audio(frame)
    await session.send_silence(600)
    await session.close()
    return [event async for event in session.events()]


async def test_the_amount_is_not_confused_with_the_day_of_the_month() -> None:
    events = await bank_turn("Bitte überweisen Sie 200 am 9. April.")

    calls = [event.tool_call for event in events if event.kind == "tool_call"]
    assert calls[0] is not None
    assert calls[0].arguments["amount"] == "200"
    assert calls[0].arguments["date"] == "2026-04-09"


async def test_it_reads_an_english_spoken_date() -> None:
    events = await bank_turn("Can you send 200 on September 4th?")

    calls = [event.tool_call for event in events if event.kind == "tool_call"]
    assert calls[0] is not None
    assert calls[0].arguments == {"amount": "200", "date": "2026-09-04"}


SHOP_TOOLS = [{"name": "check_order_status", "schema": {"order_id": "string"}}]


async def test_it_fills_an_identifier_parameter() -> None:
    adapter = MockAdapter(
        transcriber=StubTranscriber("Could you tell me the status of order number 4471?"),
        reply_ms=60,
    )
    session = await adapter.session(system_prompt=None, tools=SHOP_TOOLS)
    for frame in frames(tone(200)):
        await session.send_audio(frame)
    await session.send_silence(600)
    await session.close()

    events = [event async for event in session.events()]
    calls = [event.tool_call for event in events if event.kind == "tool_call"]
    assert calls[0] is not None
    assert calls[0].arguments == {"order_id": "4471"}
