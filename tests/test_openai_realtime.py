"""Tests for mapping OpenAI Realtime messages onto sayagain events.

The transport is not tested here (it needs a key and a network); the protocol
translation is, because that is where the bugs live.
"""

import base64

from sayagain.adapters.openai_realtime import OpenAIRealtimeAdapter, translate

REALTIME_RATE = 24_000


def test_audio_deltas_become_audio_events() -> None:
    payload = {
        "type": "response.audio.delta",
        "delta": base64.b64encode(b"\x10\x10" * 480).decode(),
    }

    event = translate(payload, t_ns=5)

    assert event is not None
    assert event.kind == "audio"
    assert event.audio is not None
    # 480 samples at 24 kHz is 20 ms, which is 320 samples at 16 kHz.
    assert len(event.audio) == 640


def test_function_calls_become_tool_calls() -> None:
    payload = {
        "type": "response.function_call_arguments.done",
        "name": "reschedule_appointment",
        "arguments": '{"date": "friday", "time": "morning"}',
    }

    event = translate(payload, t_ns=7)

    assert event is not None
    assert event.kind == "tool_call"
    assert event.tool_call is not None
    assert event.tool_call.name == "reschedule_appointment"
    assert event.tool_call.arguments == {"date": "friday", "time": "morning"}


def test_unparseable_arguments_do_not_lose_the_call() -> None:
    payload = {
        "type": "response.function_call_arguments.done",
        "name": "reschedule_appointment",
        "arguments": "{not json",
    }

    event = translate(payload, t_ns=7)

    assert event is not None
    assert event.kind == "tool_call"
    assert event.tool_call is not None
    assert event.tool_call.arguments == {}


def test_input_transcription_becomes_a_transcript() -> None:
    payload = {
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "I'd like to reschedule to Friday morning.",
    }

    event = translate(payload, t_ns=9)

    assert event is not None
    assert event.kind == "transcript"
    assert event.text == "I'd like to reschedule to Friday morning."


def test_response_done_ends_the_turn() -> None:
    event = translate({"type": "response.done"}, t_ns=11)

    assert event is not None
    assert event.kind == "end_turn"


def test_errors_are_surfaced_not_swallowed() -> None:
    event = translate({"type": "error", "error": {"message": "rate limited"}}, t_ns=13)

    assert event is not None
    assert event.kind == "error"
    assert event.text is not None
    assert "rate limited" in event.text


def test_uninteresting_messages_are_dropped() -> None:
    assert translate({"type": "session.updated"}, t_ns=1) is None
    assert translate({"type": "rate_limits.updated"}, t_ns=1) is None


def test_the_adapter_reports_its_name() -> None:
    assert OpenAIRealtimeAdapter(api_key="test").name == "openai_realtime"


def test_tools_are_translated_into_the_realtime_schema() -> None:
    adapter = OpenAIRealtimeAdapter(api_key="test")

    translated = adapter.tool_schema(
        [{"name": "reschedule_appointment", "schema": {"date": "string", "time": "string"}}]
    )

    assert translated == [
        {
            "type": "function",
            "name": "reschedule_appointment",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string"}, "time": {"type": "string"}},
                "required": ["date", "time"],
            },
        }
    ]
