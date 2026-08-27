"""Drive one case through an adapter and record everything that happened.

The runner is deliberately dumb: it speaks, perturbs, streams, listens, and
writes down what came back with timestamps. It never decides whether the agent
was right — that is `sayagain.score`, working from this log. Keeping the two
apart is what makes a failure replayable without re-running the audio.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from sayagain.adapters.base import Adapter, AgentEvent, AgentSession
from sayagain.audio import FRAME_MS, duration_ms, frames, to_float, to_pcm16, trim_silence
from sayagain.expand import Case
from sayagain.perturb import apply as apply_perturbation
from sayagain.perturb import describe
from sayagain.tts import Synthesizer

#: How long to wait for the agent to finish a turn before giving up.
DEFAULT_TIMEOUT_S = 30.0
#: Silence sent after each utterance so the agent knows the user stopped.
DEFAULT_END_SILENCE_MS = 600


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """What the user did on one turn, in session-relative nanoseconds."""

    text: str
    audio_ms: int
    user_audio_end_ns: int
    interrupt_text: str | None = None
    interrupt_start_ns: int | None = None


@dataclass(slots=True)
class RunLog:
    """Everything one case produced."""

    case_id: str
    scenario_id: str
    language: str
    register: str
    repeat: int
    perturbation: dict[str, Any]
    seed: int
    turns: list[TurnRecord] = field(default_factory=list)
    events: list[AgentEvent] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def header(self) -> dict[str, Any]:
        """Return the case metadata, written as the first line of the JSONL log."""
        return {
            "type": "case",
            "case_id": self.case_id,
            "scenario_id": self.scenario_id,
            "language": self.language,
            "register": self.register,
            "repeat": self.repeat,
            "perturbation": self.perturbation,
            "seed": self.seed,
            "error": self.error,
            "state": self.state,
            "turns": [
                {
                    "text": turn.text,
                    "audio_ms": turn.audio_ms,
                    "user_audio_end_ns": turn.user_audio_end_ns,
                    "interrupt_text": turn.interrupt_text,
                    "interrupt_start_ns": turn.interrupt_start_ns,
                }
                for turn in self.turns
            ],
        }

    def write_jsonl(self, directory: Path) -> Path:
        """Write the log so a failure can be inspected without re-running audio."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.case_id.replace('/', '_')}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.header(), ensure_ascii=False) + "\n")
            for event in self.events:
                handle.write(json.dumps(_event_json(event), ensure_ascii=False) + "\n")
        return path


class Runner:
    """Runs cases against one adapter."""

    def __init__(
        self,
        adapter: Adapter,
        synthesizer: Synthesizer,
        *,
        seed: int = 42,
        realtime: bool = True,
        end_silence_ms: int = DEFAULT_END_SILENCE_MS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        log_dir: Path | None = None,
    ) -> None:
        self.adapter = adapter
        self.synthesizer = synthesizer
        self.seed = seed
        self.realtime = realtime
        self.end_silence_ms = end_silence_ms
        self.timeout_s = timeout_s
        self.log_dir = log_dir

    async def run(self, case: Case) -> RunLog:
        """Run one case start to finish. Never raises: failures land in `RunLog.error`."""
        log = RunLog(
            case_id=case.id,
            scenario_id=case.scenario.id,
            language=case.language,
            register=case.register,
            repeat=case.repeat,
            perturbation=describe(case.perturbation),
            seed=self.seed,
        )
        session: AgentSession | None = None
        consumer: asyncio.Task[None] | None = None
        try:
            tools = [tool.model_dump(by_alias=True) for tool in case.scenario.agent.tools]
            session = await self.adapter.session(
                system_prompt=case.scenario.system_prompt(), tools=tools
            )
            consumer = asyncio.create_task(_collect(session, log.events))
            await self._play(case, session, log)
            log.state = await session.state()
        except Exception as error:
            log.error = f"{type(error).__name__}: {error}"
        finally:
            if session is not None:
                with contextlib.suppress(Exception):
                    await session.close()
            if consumer is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(consumer, timeout=self.timeout_s)

        if self.log_dir is not None:
            log.write_jsonl(self.log_dir)
        return log

    async def _play(self, case: Case, session: AgentSession, log: RunLog) -> None:
        started = time.perf_counter_ns()
        expected_end_turns = 0
        for index, text in enumerate(case.turn_texts):
            pcm = await self._prepare(text, case)
            await self._stream(session, pcm)
            record = TurnRecord(
                text=text,
                audio_ms=duration_ms(pcm),
                user_audio_end_ns=time.perf_counter_ns() - started,
            )
            # The trailing silence is what tells the agent the user stopped, so it
            # has to go out before there is anything to interrupt.
            await session.send_silence(self.end_silence_ms)

            interrupt_text = case.interrupt_texts[index]
            spec = case.scenario.turns[index].interrupt
            if interrupt_text is not None and spec is not None:
                start = await self._interrupt(
                    case, session, log, interrupt_text, spec.after_agent_speaks_ms, started
                )
                if start is not None:
                    record = replace(
                        record, interrupt_text=interrupt_text, interrupt_start_ns=start
                    )
                    await session.send_silence(self.end_silence_ms)
                    expected_end_turns += 1  # the reply that got cut off

            expected_end_turns += 1
            await self._await_end_of_turn(log, expected=expected_end_turns)
            log.turns.append(record)

    async def _prepare(self, text: str, case: Case) -> bytes:
        """Speak the line, then damage it exactly the way the case says."""
        voices = case.scenario.matrix.voices_for(case.language)
        pcm = await self.synthesizer.say(
            text, language=case.language, voice=voices[0] if voices else None
        )
        # Trim before perturbing, so the noise bed is not trimmed with it. The
        # silence TTS adds at either end is not something the caller said, and
        # leaving it in ends the agent's turn before the audio has arrived.
        damaged = apply_perturbation(to_float(trim_silence(pcm)), case.perturbation, seed=self.seed)
        return to_pcm16(damaged)

    async def _stream(self, session: AgentSession, pcm: bytes) -> None:
        for frame in frames(pcm):
            await session.send_audio(frame)
            if self.realtime:
                await asyncio.sleep(FRAME_MS / 1000)

    async def _interrupt(
        self,
        case: Case,
        session: AgentSession,
        log: RunLog,
        text: str,
        after_ms: int,
        started: int,
    ) -> int | None:
        """Wait for the agent to get talking, then talk over it.

        If it never speaks there is nothing to barge into, so the interruption
        is skipped rather than fired into silence.
        """
        spoke = await _wait_for(
            lambda: any(event.kind == "audio" for event in log.events), self.timeout_s
        )
        if not spoke:
            return None
        await asyncio.sleep(after_ms / 1000)
        pcm = await self._prepare(text, case)
        start = time.perf_counter_ns() - started
        await self._stream(session, pcm)
        return start

    async def _await_end_of_turn(self, log: RunLog, *, expected: int) -> None:
        await _wait_for(
            lambda: sum(1 for e in log.events if e.kind == "end_turn") >= expected,
            self.timeout_s,
        )


async def _collect(session: AgentSession, sink: list[AgentEvent]) -> None:
    async for event in session.events():
        sink.append(event)


async def _wait_for(predicate: Callable[[], bool], timeout_s: float) -> bool:
    """Poll until `predicate()` is true, or give up."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return False


def _event_json(event: AgentEvent) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": event.kind, "t_ns": event.t_ns}
    if event.text is not None:
        payload["text"] = event.text
    if event.tool_call is not None:
        payload["tool_call"] = {
            "name": event.tool_call.name,
            "arguments": event.tool_call.arguments,
            "t_ns": event.tool_call.t_ns,
        }
    if event.audio is not None:
        # Audio bytes stay out of the log; the report needs sizes and timing, not waveforms.
        payload["audio_bytes"] = len(event.audio)
    return payload
