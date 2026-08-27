"""An in-process toy agent, used by the tests and the README demo.

It is deliberately shallow: transcribe, match a few keywords, call one tool,
speak back. That shallowness is the point — it degrades the way a real cascade
agent degrades once the audio gets hard, so the harness has something honest to
measure without anyone needing an API key.

The keyword tables here are a placeholder for `sayagain.normalize`, which grows
the real cross-language vocabulary on day 3.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import AsyncIterator
from typing import Any, Protocol

from sayagain.adapters.base import AgentEvent, Clock, ToolCall
from sayagain.audio import FRAME_MS, frames, has_speech, silence, tone

#: Canonical weekday -> the surface forms the mock recognises.
WEEKDAYS: dict[str, set[str]] = {
    "monday": {"monday", "segunda", "lunes", "montag"},
    "tuesday": {"tuesday", "terca", "martes", "dienstag"},
    "wednesday": {"wednesday", "quarta", "miercoles", "mittwoch"},
    "thursday": {"thursday", "quinta", "jueves", "donnerstag"},
    "friday": {"friday", "sexta", "viernes", "freitag"},
    "saturday": {"saturday", "sabado", "samstag"},
    "sunday": {"sunday", "domingo", "sonntag"},
}

#: Canonical part of day -> the surface forms the mock recognises.
DAYPARTS: dict[str, set[str]] = {
    "morning": {"morning", "manha", "manana", "morgen", "vormittag"},
    "afternoon": {"afternoon", "tarde", "nachmittag"},
    "evening": {"evening", "night", "noite", "noche", "abend"},
}

#: Tool-name tokens -> other words that should trigger the same tool.
SYNONYMS: dict[str, set[str]] = {
    "reschedule": {
        "move",
        "change",
        "shift",
        "reagendar",
        "remarcar",
        "mudar",
        "jogar",
        "mover",
        "cambiar",
        "verschieben",
    },
    "appointment": {"consulta", "cita", "termin", "agendamento", "appointments"},
    "cancel": {"cancelar", "stornieren"},
    "order": {"pedido", "orden", "bestellung"},
    "status": {"estado", "situacao"},
    "transfer": {"transferir", "ueberweisen"},
    "money": {"dinheiro", "dinero", "geld"},
}

_WEEKDAY_BY_FORM = {form: day for day, forms in WEEKDAYS.items() for form in forms}
_DAYPART_BY_FORM = {form: part for part, forms in DAYPARTS.items() for form in forms}

#: Tool argument name -> the state field it moves, for `state()`.
STATE_FIELDS = {"date": "day", "time": "time"}


def normalise(text: str) -> str:
    """Lowercase and strip accents so `manhã` and `manha` are one word."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def tokens(text: str) -> list[str]:
    """Split text into accent-free alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", normalise(text))


class Transcriber(Protocol):
    """Turns user audio into text. Injected so tests need no model."""

    def transcribe(self, pcm: bytes, language: str | None = None) -> str:
        """Transcribe 16 kHz mono PCM16 audio."""
        ...


class WhisperTranscriber:
    """faster-whisper, pinned as hard as the library allows.

    Greedy decoding with a fixed seed is the closest CTranslate2 gets to
    reproducible output; it is not a guarantee, so nothing in the scoring path
    should treat this transcript as ground truth.
    """

    def __init__(self, model_size: str = "small", *, seed: int = 42) -> None:
        self.model_size = model_size
        self.seed = seed
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            import ctranslate2
            from faster_whisper import WhisperModel

            ctranslate2.set_random_seed(self.seed)
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, pcm: bytes, language: str | None = None) -> str:
        """Transcribe with beam_size=1 and temperature=0 for the least variance."""
        from sayagain.audio import to_float

        segments, _ = self._load().transcribe(
            to_float(pcm).astype("float32"),
            language=language.split("-")[0] if language else None,
            beam_size=1,
            temperature=0.0,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class MockSession:
    """One conversation with the toy agent."""

    def __init__(
        self,
        transcriber: Transcriber,
        tools: list[dict[str, Any]],
        *,
        end_of_turn_silence_ms: int = 400,
        reply_ms: int = 1500,
    ) -> None:
        self._transcriber = transcriber
        self._tools = tools
        self._end_of_turn_silence_ms = end_of_turn_silence_ms
        self._reply_ms = reply_ms
        self._clock = Clock()
        self._queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._buffer = bytearray()
        self._silence_ms = 0
        self._heard_speech = False
        self._responded = False
        self._closed = False
        self._speaking = False
        self._interrupted = False
        self._speech: asyncio.Task[None] | None = None
        self._response: asyncio.Task[None] | None = None
        self._turn_id = 0
        self._state: dict[str, Any] = {}

    async def send_audio(self, frame: bytes) -> None:
        """Accept one 20 ms frame of user audio, and yield the floor if talked over."""
        if has_speech(frame):
            if self._responded:
                # Speech after this agent has already answered starts a new turn.
                # If it is still talking, that turn is a barge-in and it shuts up.
                if self._speaking:
                    self._interrupted = True
                self._start_new_turn()
            self._heard_speech = True
            self._silence_ms = 0
        else:
            self._silence_ms += FRAME_MS
        self._buffer.extend(frame)
        await self._maybe_respond()

    async def send_silence(self, ms: int) -> None:
        """Accept a gap, which is how this agent learns the user stopped."""
        self._silence_ms += ms
        self._buffer.extend(silence(ms))
        await self._maybe_respond()

    async def events(self) -> AsyncIterator[AgentEvent]:
        """Yield events until the session is closed."""
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def state(self) -> dict[str, Any]:
        """Report what the agent believes it changed."""
        return dict(self._state)

    async def close(self) -> None:
        """End the event stream. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        # Let the reply finish rather than truncating it: closing means the turn
        # is over, and a caller that wants it cut short interrupts instead.
        if self._response is not None:
            await self._response
        if self._speech is not None:
            await self._speech
        await self._queue.put(None)

    def _start_new_turn(self) -> None:
        """Forget the previous utterance and listen to the new one."""
        self._turn_id += 1
        self._buffer = bytearray()
        self._responded = False
        self._heard_speech = False
        self._silence_ms = 0

    async def _maybe_respond(self) -> None:
        if self._responded or self._speaking or not self._heard_speech:
            return
        if self._silence_ms < self._end_of_turn_silence_ms:
            return
        self._responded = True
        # Answering happens off to the side: a real agent is another process, and
        # blocking the caller here would stall the audio still being streamed in.
        self._response = asyncio.create_task(self._respond())

    async def _respond(self) -> None:
        turn = self._turn_id
        text = await asyncio.to_thread(self._transcriber.transcribe, bytes(self._buffer))
        if turn != self._turn_id:
            return  # interrupted while transcribing; that reply is stale now
        await self._emit("transcript", text=text)

        call = self._decide(text)
        if call is not None:
            await self._emit("tool_call", tool_call=call)
            self._state.update(_state_updates(call.name, call.arguments))

        self._speaking = True
        self._interrupted = False
        self._speech = asyncio.create_task(self._speak())

    async def _speak(self) -> None:
        """Emit the reply in real time, so an interruption can actually cut it off."""
        try:
            for frame in frames(tone(self._reply_ms)):
                if self._interrupted:
                    break
                await self._emit("audio", audio=frame)
                await asyncio.sleep(FRAME_MS / 1000)
            await self._emit("end_turn")
        finally:
            self._speaking = False
        # A correction can arrive mid-reply, and while it does nothing else will
        # call _maybe_respond, so answering it has to be triggered from here.
        await self._maybe_respond()

    def _decide(self, text: str) -> ToolCall | None:
        heard = tokens(text)
        for tool in self._tools:
            name = str(tool["name"])
            if not _tool_matches(name, heard):
                continue
            schema = tool.get("schema") or {}
            arguments = _extract_arguments(heard, list(schema))
            return ToolCall(name=name, arguments=arguments, t_ns=self._clock.now())
        return None

    async def _emit(
        self,
        kind: Any,
        *,
        audio: bytes | None = None,
        tool_call: ToolCall | None = None,
        text: str | None = None,
    ) -> None:
        event = AgentEvent(
            kind=kind, t_ns=self._clock.now(), audio=audio, tool_call=tool_call, text=text
        )
        await self._queue.put(event)


class MockAdapter:
    """Adapter for the in-process toy agent."""

    name = "mock"

    def __init__(
        self,
        *,
        transcriber: Transcriber | None = None,
        end_of_turn_silence_ms: int = 400,
        reply_ms: int = 1500,
    ) -> None:
        self._transcriber = transcriber
        self._end_of_turn_silence_ms = end_of_turn_silence_ms
        self._reply_ms = reply_ms

    async def session(
        self, *, system_prompt: str | None = None, tools: list[dict[str, Any]]
    ) -> MockSession:
        """Open a fresh toy-agent session."""
        return MockSession(
            self._transcriber or WhisperTranscriber(),
            tools,
            end_of_turn_silence_ms=self._end_of_turn_silence_ms,
            reply_ms=self._reply_ms,
        )


def _tool_matches(tool_name: str, heard: list[str]) -> bool:
    spoken = set(heard)
    for part in tool_name.split("_"):
        if ({part} | SYNONYMS.get(part, set())) & spoken:
            return True
    return False


def _extract_arguments(heard: list[str], parameters: list[str]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for parameter in parameters:
        if parameter in {"date", "day"}:
            value = _first_match(heard, _WEEKDAY_BY_FORM)
        elif parameter == "time":
            value = _first_match(heard, _DAYPART_BY_FORM)
        else:
            value = None
        if value is not None:
            found[parameter] = value
    return found


def _first_match(heard: list[str], table: dict[str, str]) -> str | None:
    for token in heard:
        if token in table:
            return table[token]
    return None


def _state_updates(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    namespace = tool_name.split("_")[-1]
    return {f"{namespace}.{STATE_FIELDS.get(key, key)}": value for key, value in arguments.items()}
