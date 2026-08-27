"""A thin wrapper over the OpenAI Realtime API.

Two jobs, kept apart on purpose: `translate` maps one Realtime message onto one
`AgentEvent` and is a pure function you can test without a key, and the session
class owns the websocket. Realtime speaks 24 kHz PCM16, so this file is also
where that gets converted to the 16 kHz everything above the adapter expects.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import AsyncIterator
from typing import Any

from sayagain.adapters.base import AgentEvent, Clock, ToolCall
from sayagain.audio import SAMPLE_RATE, resample_pcm16

#: The rate the Realtime API speaks, both directions.
REALTIME_RATE = 24_000
REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime"


class OpenAIRealtimeError(Exception):
    """The Realtime session could not be established or maintained."""


def translate(payload: dict[str, Any], *, t_ns: int) -> AgentEvent | None:
    """Map one Realtime message onto an `AgentEvent`, or None if it says nothing.

    Most of the Realtime stream is bookkeeping. Only the six kinds below carry
    anything the harness measures.
    """
    kind = payload.get("type")

    if kind == "response.audio.delta":
        raw = base64.b64decode(payload.get("delta", ""))
        return AgentEvent(
            kind="audio",
            t_ns=t_ns,
            audio=resample_pcm16(raw, from_rate=REALTIME_RATE, to_rate=SAMPLE_RATE),
        )

    if kind == "response.function_call_arguments.done":
        return AgentEvent(
            kind="tool_call",
            t_ns=t_ns,
            tool_call=ToolCall(
                name=str(payload.get("name", "")),
                arguments=_loads(payload.get("arguments")),
                t_ns=t_ns,
            ),
        )

    if kind == "conversation.item.input_audio_transcription.completed":
        return AgentEvent(kind="transcript", t_ns=t_ns, text=payload.get("transcript"))

    if kind in {"response.done", "response.audio.done"}:
        return AgentEvent(kind="end_turn", t_ns=t_ns)

    if kind == "error":
        message = payload.get("error", {})
        return AgentEvent(kind="error", t_ns=t_ns, text=str(message.get("message", message)))

    return None


class OpenAIRealtimeSession:
    """One Realtime websocket, dressed as an `AgentSession`."""

    def __init__(self, connection: Any, clock: Clock) -> None:
        self._connection = connection
        self._clock = clock
        self._state: dict[str, Any] = {}
        self._closed = False

    async def send_audio(self, frame: bytes) -> None:
        """Append one 20 ms frame, converted up to the rate Realtime wants."""
        upsampled = resample_pcm16(frame, from_rate=SAMPLE_RATE, to_rate=REALTIME_RATE)
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(upsampled).decode(),
            }
        )

    async def send_silence(self, ms: int) -> None:
        """Send silence so server-side VAD sees the user stop."""
        quiet = b"\x00" * (REALTIME_RATE * ms // 1000 * 2)
        await self._send(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(quiet).decode()}
        )

    async def events(self) -> AsyncIterator[AgentEvent]:
        """Yield translated events until the socket closes."""
        try:
            async for message in self._connection:
                payload = json.loads(message)
                event = translate(payload, t_ns=self._clock.now())
                if event is None:
                    continue
                if event.tool_call is not None:
                    self._state[event.tool_call.name] = event.tool_call.arguments
                yield event
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        except Exception as error:
            yield AgentEvent(kind="error", t_ns=self._clock.now(), text=str(error))

    async def state(self) -> dict[str, Any]:
        """Realtime reports no application state, so this is what it called."""
        return dict(self._state)

    async def close(self) -> None:
        """Close the socket. Safe to call twice."""
        if not self._closed:
            self._closed = True
            await self._connection.close()

    async def _send(self, payload: dict[str, Any]) -> None:
        await self._connection.send(json.dumps(payload))


class OpenAIRealtimeAdapter:
    """Connects to OpenAI's Realtime API with server-side turn detection."""

    name = "openai_realtime"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        voice: str = "alloy",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.voice = voice

    def tool_schema(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert sayagain's compact tool declaration into Realtime's schema."""
        translated = []
        for tool in tools:
            properties = {name: {"type": kind} for name, kind in (tool.get("schema") or {}).items()}
            translated.append(
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties),
                    },
                }
            )
        return translated

    async def session(
        self, *, system_prompt: str | None = None, tools: list[dict[str, Any]]
    ) -> OpenAIRealtimeSession:
        """Open a Realtime session configured for server VAD.

        Raises:
            OpenAIRealtimeError: no API key, or the socket refused to open.

        """
        if not self.api_key:
            raise OpenAIRealtimeError("OPENAI_API_KEY is not set")
        try:
            from websockets.asyncio.client import connect
        except ImportError as error:  # pragma: no cover - websockets is a hard dependency
            raise OpenAIRealtimeError("websockets is not installed") from error

        connection = await connect(
            f"{REALTIME_URL}?model={self.model}",
            additional_headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        )
        await connection.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"],
                        "instructions": system_prompt or "",
                        "voice": self.voice,
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {"model": "whisper-1"},
                        "turn_detection": {"type": "server_vad"},
                        "tools": self.tool_schema(tools),
                    },
                }
            )
        )
        # The clock starts only once the socket is configured: session setup is
        # not the agent thinking, and counting it would inflate every latency.
        return OpenAIRealtimeSession(connection, Clock())


def _loads(raw: Any) -> dict[str, Any]:
    """Parse tool arguments, keeping the call even when the JSON is broken."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
