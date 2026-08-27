"""Tests for the Adapter / AgentSession contract."""

from collections.abc import AsyncIterator

from sayagain.adapters.base import Adapter, AgentEvent, AgentSession, ToolCall


class ConformingSession:
    async def send_audio(self, frame: bytes) -> None: ...

    async def send_silence(self, ms: int) -> None: ...

    def events(self) -> AsyncIterator[AgentEvent]:  # pragma: no cover - shape only
        raise NotImplementedError

    async def state(self) -> dict[str, object]:
        return {}

    async def close(self) -> None: ...


class ConformingAdapter:
    name = "conforming"

    async def session(
        self, *, system_prompt: str | None, tools: list[dict[str, object]]
    ) -> ConformingSession:
        return ConformingSession()


class NotASession:
    async def send_audio(self, frame: bytes) -> None: ...


def test_a_conforming_session_satisfies_the_protocol() -> None:
    assert isinstance(ConformingSession(), AgentSession)


def test_a_partial_implementation_does_not_satisfy_the_protocol() -> None:
    assert not isinstance(NotASession(), AgentSession)


def test_a_conforming_adapter_satisfies_the_protocol() -> None:
    assert isinstance(ConformingAdapter(), Adapter)


def test_agent_event_defaults_are_empty() -> None:
    event = AgentEvent(kind="end_turn", t_ns=123)

    assert event.audio is None
    assert event.tool_call is None
    assert event.text is None


def test_tool_call_carries_its_own_timestamp() -> None:
    call = ToolCall(name="reschedule_appointment", arguments={"date": "friday"}, t_ns=42)

    assert call.name == "reschedule_appointment"
    assert call.arguments == {"date": "friday"}
    assert call.t_ns == 42
