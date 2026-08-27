"""The Adapter / AgentSession contract every backend implements.

This is the seam between sayagain and the agent under test. Everything above it
speaks one format: 16 kHz mono PCM16 in 20 ms frames, and timestamps in
nanoseconds relative to the start of the session. An adapter converts whatever
its backend actually speaks into that, at its own edge.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

EventKind = Literal["audio", "tool_call", "transcript", "end_turn", "error"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool the agent decided to call."""

    name: str
    arguments: dict[str, Any]
    t_ns: int


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """Anything the agent did, stamped relative to the start of the session."""

    kind: EventKind
    t_ns: int
    audio: bytes | None = None
    tool_call: ToolCall | None = None
    text: str | None = None


@runtime_checkable
class AgentSession(Protocol):
    """One conversation with the agent under test."""

    async def send_audio(self, frame: bytes) -> None:
        """Send one 20 ms frame of user audio."""
        ...

    async def send_silence(self, ms: int) -> None:
        """Send `ms` of silence, which is how the agent learns the user stopped."""
        ...

    # Implementations are async generators, so this is deliberately not `async def`:
    # an async generator function returns an AsyncIterator, it does not await one.
    def events(self) -> AsyncIterator[AgentEvent]:
        """Yield agent events until the session closes."""
        ...

    async def state(self) -> dict[str, Any]:
        """Agent-reported end state, keyed by dotted path. May be empty."""
        ...

    async def close(self) -> None:
        """Release the session; safe to call twice."""
        ...


@runtime_checkable
class Adapter(Protocol):
    """A way of connecting to one kind of voice agent."""

    name: str

    async def session(
        self, *, system_prompt: str | None, tools: list[dict[str, Any]]
    ) -> AgentSession:
        """Open a fresh session; each case gets its own."""
        ...


class Clock:
    """Session-relative nanosecond timestamps, as required of every event."""

    def __init__(self) -> None:
        self._start = time.perf_counter_ns()

    def now(self) -> int:
        """Nanoseconds elapsed since this clock was created."""
        return time.perf_counter_ns() - self._start
