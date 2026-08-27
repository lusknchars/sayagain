"""The generic WebSocket adapter: make any agent testable in about an hour.

The wire protocol is deliberately tiny and is documented in full in
`docs/adapter-protocol.md`. Audio goes both ways as raw binary frames — 16 kHz
mono PCM16, 20 ms, 640 bytes — and everything else is a small JSON object with
a `type`. There is no framing, no handshake beyond one `start` message, and no
dependency on any particular language or framework on the other end.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from sayagain.adapters.base import AgentEvent, Clock, ToolCall
from sayagain.audio import FRAME_MS, SAMPLE_RATE


class WebSocketAdapterError(Exception):
    """The agent socket could not be reached or spoke nonsense."""


def translate(message: str | bytes, *, t_ns: int) -> AgentEvent | None:
    """Map one incoming websocket message onto an `AgentEvent`.

    Binary is always agent audio. Text is always JSON with a `type`. Anything
    unrecognised is ignored rather than fatal, so an agent may send its own
    bookkeeping over the same socket.
    """
    if isinstance(message, bytes | bytearray):
        return AgentEvent(kind="audio", t_ns=t_ns, audio=bytes(message))

    try:
        payload = json.loads(message)
    except json.JSONDecodeError as error:
        return AgentEvent(kind="error", t_ns=t_ns, text=f"unparseable message: {error}")
    if not isinstance(payload, dict):
        return AgentEvent(kind="error", t_ns=t_ns, text="expected a JSON object")

    kind = payload.get("type")
    if kind == "tool_call":
        return AgentEvent(
            kind="tool_call",
            t_ns=t_ns,
            tool_call=ToolCall(
                name=str(payload.get("name", "")),
                arguments=payload.get("arguments") or {},
                t_ns=t_ns,
            ),
        )
    if kind == "transcript":
        return AgentEvent(kind="transcript", t_ns=t_ns, text=payload.get("text"))
    if kind == "end_turn":
        return AgentEvent(kind="end_turn", t_ns=t_ns)
    if kind == "error":
        return AgentEvent(kind="error", t_ns=t_ns, text=str(payload.get("message", "")))
    # `state` is folded into the session rather than emitted as an event.
    return None


class WebSocketSession:
    """One conversation over one socket."""

    def __init__(self, connection: Any, clock: Clock | None = None) -> None:
        self._connection = connection
        self._clock = clock or Clock()
        self._state: dict[str, Any] = {}
        self._closed = False

    async def send_audio(self, frame: bytes) -> None:
        """Send one 20 ms PCM16 frame as a binary message."""
        await self._connection.send(frame)

    async def send_silence(self, ms: int) -> None:
        """Tell the agent the user went quiet, without shipping the zeros."""
        await self._connection.send(json.dumps({"type": "silence", "ms": ms}))

    async def events(self) -> AsyncIterator[AgentEvent]:
        """Yield events until the agent closes the socket."""
        try:
            async for message in self._connection:
                if isinstance(message, str):
                    self._absorb_state(message)
                event = translate(message, t_ns=self._clock.now())
                if event is not None:
                    yield event
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        except Exception as error:
            yield AgentEvent(kind="error", t_ns=self._clock.now(), text=str(error))

    async def state(self) -> dict[str, Any]:
        """Whatever the agent last reported in a `state` message."""
        return dict(self._state)

    async def close(self) -> None:
        """Close the socket. Safe to call twice."""
        if not self._closed:
            self._closed = True
            await self._connection.close()

    def _absorb_state(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict) and payload.get("type") == "state":
            reported = payload.get("state")
            if isinstance(reported, dict):
                self._state.update(reported)


class WebSocketAdapter:
    """Talks to any agent that implements the sayagain wire protocol."""

    name = "websocket"

    def __init__(self, url: str) -> None:
        self.url = url

    async def start(
        self,
        connection: Any,
        *,
        system_prompt: str | None,
        tools: list[dict[str, Any]],
    ) -> WebSocketSession:
        """Send the opening `start` message and wrap the socket in a session."""
        await connection.send(
            json.dumps(
                {
                    "type": "start",
                    "system_prompt": system_prompt,
                    "tools": tools,
                    "sample_rate": SAMPLE_RATE,
                    "frame_ms": FRAME_MS,
                }
            )
        )
        return WebSocketSession(connection)

    async def session(
        self, *, system_prompt: str | None = None, tools: list[dict[str, Any]]
    ) -> WebSocketSession:
        """Connect to the agent and announce the session.

        Raises:
            WebSocketAdapterError: the socket could not be opened.

        """
        try:
            from websockets.asyncio.client import connect
        except ImportError as error:  # pragma: no cover - websockets is a hard dependency
            raise WebSocketAdapterError("websockets is not installed") from error
        try:
            connection = await connect(self.url)
        except Exception as error:
            raise WebSocketAdapterError(f"could not connect to {self.url}: {error}") from error
        return await self.start(connection, system_prompt=system_prompt, tools=tools)
