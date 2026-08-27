"""Tests for the generic WebSocket adapter, against a fake socket."""

import json

from sayagain.adapters.base import AgentEvent
from sayagain.adapters.websocket import WebSocketAdapter, WebSocketSession, translate
from sayagain.audio import FRAME_BYTES


class FakeConnection:
    """A websocket that records what was sent and replays what we scripted."""

    def __init__(self, incoming: list[str | bytes] | None = None) -> None:
        self.sent: list[str | bytes] = []
        self.incoming = incoming or []
        self.closed = False

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> "FakeConnection":
        self._iter = iter(self.incoming)
        return self

    async def __anext__(self) -> str | bytes:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


def test_binary_frames_are_agent_audio() -> None:
    event = translate(b"\x10\x10" * 320, t_ns=1)

    assert event is not None
    assert event.kind == "audio"
    assert event.audio is not None
    assert len(event.audio) == FRAME_BYTES


def test_tool_calls_arrive_as_json() -> None:
    payload = json.dumps(
        {"type": "tool_call", "name": "reschedule_appointment", "arguments": {"date": "friday"}}
    )

    event = translate(payload, t_ns=2)

    assert event is not None
    assert event.tool_call is not None
    assert event.tool_call.name == "reschedule_appointment"
    assert event.tool_call.arguments == {"date": "friday"}


def test_transcripts_and_end_of_turn() -> None:
    transcript = translate(json.dumps({"type": "transcript", "text": "hello"}), t_ns=3)
    end = translate(json.dumps({"type": "end_turn"}), t_ns=4)

    assert transcript is not None
    assert transcript.text == "hello"
    assert end is not None
    assert end.kind == "end_turn"


def test_state_messages_are_not_events() -> None:
    assert translate(json.dumps({"type": "state", "state": {"a.b": 1}}), t_ns=5) is None


def test_unknown_message_types_are_ignored() -> None:
    assert translate(json.dumps({"type": "heartbeat"}), t_ns=6) is None


def test_malformed_json_surfaces_as_an_error() -> None:
    event = translate("{not json", t_ns=7)

    assert event is not None
    assert event.kind == "error"


async def test_the_session_announces_itself_on_connect() -> None:
    connection = FakeConnection()
    adapter = WebSocketAdapter("ws://localhost:9999/agent")

    await adapter.start(connection, system_prompt="be helpful", tools=[{"name": "t", "schema": {}}])

    start = json.loads(connection.sent[0])
    assert start["type"] == "start"
    assert start["system_prompt"] == "be helpful"
    assert start["sample_rate"] == 16_000
    assert start["frame_ms"] == 20
    assert start["tools"] == [{"name": "t", "schema": {}}]


async def test_audio_goes_out_as_binary_and_silence_as_json() -> None:
    connection = FakeConnection()
    session = WebSocketSession(connection)

    await session.send_audio(b"\x00\x01" * 320)
    await session.send_silence(400)

    assert connection.sent[0] == b"\x00\x01" * 320
    assert json.loads(connection.sent[1]) == {"type": "silence", "ms": 400}


async def test_the_session_collects_state_while_it_reads_events() -> None:
    connection = FakeConnection(
        [
            json.dumps({"type": "transcript", "text": "move it to friday"}),
            json.dumps({"type": "state", "state": {"appointment.day": "friday"}}),
            json.dumps({"type": "end_turn"}),
        ]
    )
    session = WebSocketSession(connection)

    events: list[AgentEvent] = [event async for event in session.events()]

    assert [event.kind for event in events] == ["transcript", "end_turn"]
    assert await session.state() == {"appointment.day": "friday"}


async def test_closing_closes_the_socket() -> None:
    connection = FakeConnection()
    session = WebSocketSession(connection)

    await session.close()
    await session.close()

    assert connection.closed is True
