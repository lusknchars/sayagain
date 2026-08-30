r"""A runnable reference agent for the sayagain WebSocket protocol.

This is the file to copy when wiring up your own agent. It is deliberately
standalone — it imports nothing from `sayagain` — so it demonstrates that the
protocol needs no library on the agent side. Everything it does could be done
in any language in about an hour.

Run it in one terminal:

    uv run python examples/agents/local_agent.py

Point sayagain at it in another:

    uv run sayagain run examples/reschedule_appointment.yaml \\
        --adapter websocket --url ws://localhost:8765 \\
        --only-language en-US --repeats 1

It transcribes with faster-whisper when that is installed, and falls back to a
canned line when it is not, so the plumbing can be exercised on a machine with
no model downloaded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import struct
import sys
from collections.abc import Callable
from typing import Any

from websockets.asyncio.server import serve

RATE = 16_000
FRAME_SAMPLES = 320
FRAME_BYTES = FRAME_SAMPLES * 2

WEEKDAYS = {
    "monday": {"monday", "mon", "segunda", "lunes", "montag"},
    "tuesday": {"tuesday", "tue", "terca", "martes", "dienstag"},
    "wednesday": {"wednesday", "wed", "quarta", "miercoles", "mittwoch"},
    "thursday": {"thursday", "thu", "quinta", "jueves", "donnerstag"},
    "friday": {"friday", "fri", "sexta", "viernes", "freitag"},
    "saturday": {"saturday", "sat", "sabado", "samstag"},
    "sunday": {"sunday", "sun", "domingo", "sonntag"},
}
DAYPARTS = {
    "morning": {"morning", "manha", "manana", "morgen"},
    "afternoon": {"afternoon", "tarde", "nachmittag"},
    "evening": {"evening", "night", "noite", "noche", "abend"},
}
TRIGGERS = {"reschedule", "move", "change", "shift", "appointment", "reagendar", "mudar", "jogar"}


def words(text: str) -> list[str]:
    """Split text into lowercase alphanumeric words."""
    return re.findall(r"[a-z0-9]+", text.lower())


def lookup(table: dict[str, set[str]], spoken: list[str]) -> str | None:
    """Return the first canonical value any spoken word maps to."""
    for word in spoken:
        for canonical, forms in table.items():
            if word in forms:
                return canonical
    return None


def tone_frames(ms: int, hz: float = 200.0) -> list[bytes]:
    """Stand-in speech: a sine at 16 kHz, cut into 20 ms frames."""
    frames = []
    for index in range(ms // 20):
        samples = [
            int(8000 * math.sin(2 * math.pi * hz * (index * FRAME_SAMPLES + n) / RATE))
            for n in range(FRAME_SAMPLES)
        ]
        frames.append(struct.pack(f"<{FRAME_SAMPLES}h", *samples))
    return frames


class Transcriber:
    """faster-whisper if it is installed, a canned line if it is not."""

    def __init__(self) -> None:
        self._model: Any | None = None
        try:
            import faster_whisper  # noqa: F401

            self.available = True
        except ImportError:
            self.available = False

    def __call__(self, pcm: bytes) -> str:
        """Transcribe 16 kHz mono PCM16 audio."""
        if not self.available:
            return "I would like to reschedule my appointment to Friday morning."
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel("small", device="cpu", compute_type="int8")
        import numpy as np

        audio = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        segments, _ = self._model.transcribe(audio, beam_size=1, temperature=0.0)
        return " ".join(segment.text.strip() for segment in segments).strip()


class Session:
    """One caller. Buffers audio, answers after a pause, stops when talked over."""

    def __init__(self, socket: Any, transcribe: Callable[[bytes], str], reply_ms: int) -> None:
        self.socket = socket
        self.transcribe = transcribe
        self.reply_ms = reply_ms
        self.tools: list[dict[str, Any]] = []
        self.buffer = bytearray()
        self.heard_speech = False
        self.answering = False
        self.interrupted = False
        self.task: asyncio.Task[None] | None = None

    async def handle(self, message: str | bytes) -> None:
        """Route one incoming websocket message."""
        if isinstance(message, bytes | bytearray):
            if self.answering:
                self.interrupted = True  # the caller is talking over us
            self.buffer.extend(message)
            self.heard_speech = True
            return

        event = json.loads(message)
        if event.get("type") == "start":
            self.tools = event.get("tools") or []
            print(f"  session started, {len(self.tools)} tool(s)", flush=True)
        elif event.get("type") == "silence" and event.get("ms", 0) >= 400:
            if self.heard_speech and not self.answering:
                # Answer on a task, not inline. Awaiting it here would stop this
                # loop reading the socket, and an agent that is not reading
                # cannot notice it is being talked over.
                self.answering = True
                self.task = asyncio.create_task(self.answer())

    async def answer(self) -> None:
        """Transcribe what was said, call a tool, speak, and end the turn."""
        self.interrupted = False
        heard = await asyncio.to_thread(self.transcribe, bytes(self.buffer))
        self.buffer = bytearray()
        self.heard_speech = False
        print(f"  heard: {heard!r}", flush=True)
        await self.send({"type": "transcript", "text": heard})

        call = self.decide(heard)
        if call is not None:
            print(f"  calling: {call['name']}{call['arguments']}", flush=True)
            await self.send({"type": "tool_call", **call})
            await self.send({"type": "state", "state": self.state(call)})

        for frame in tone_frames(self.reply_ms):
            if self.interrupted:
                print("  interrupted, stopping", flush=True)
                break
            await self.socket.send(frame)
            await asyncio.sleep(0.02)
        await self.send({"type": "end_turn"})
        self.answering = False

    def decide(self, heard: str) -> dict[str, Any] | None:
        """Pick a tool and fill whatever arguments the sentence names."""
        spoken = words(heard)
        if not TRIGGERS & set(spoken) or not self.tools:
            return None
        arguments: dict[str, Any] = {}
        for parameter in self.tools[0].get("schema") or {}:
            if parameter == "date":
                value = lookup(WEEKDAYS, spoken)
            elif parameter == "time":
                value = lookup(DAYPARTS, spoken)
            else:
                value = None
            if value is not None:
                arguments[parameter] = value
        return {"name": self.tools[0]["name"], "arguments": arguments}

    def state(self, call: dict[str, Any]) -> dict[str, Any]:
        """Report the effect of a tool call as dotted paths."""
        namespace = call["name"].split("_")[-1]
        rename = {"date": "day"}
        return {f"{namespace}.{rename.get(k, k)}": v for k, v in call["arguments"].items()}

    async def send(self, payload: dict[str, Any]) -> None:
        """Send one JSON control message."""
        await self.socket.send(json.dumps(payload))


async def serve_forever(host: str, port: int, reply_ms: int) -> None:
    """Accept callers until interrupted."""
    transcriber = Transcriber()
    if not transcriber.available:
        print("faster-whisper not installed: replying from a canned line", file=sys.stderr)

    async def handler(socket: Any) -> None:
        print("caller connected", flush=True)
        session = Session(socket, transcriber, reply_ms)
        try:
            async for message in socket:
                await session.handle(message)
        finally:
            if session.task is not None:
                await asyncio.gather(session.task, return_exceptions=True)
            print("caller gone", flush=True)

    async with serve(handler, host, port):
        print(f"listening on ws://{host}:{port}", flush=True)
        await asyncio.Future()


def main() -> None:
    """Parse arguments and serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reply-ms", type=int, default=1500)
    arguments = parser.parse_args()
    try:
        asyncio.run(serve_forever(arguments.host, arguments.port, arguments.reply_ms))
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()
