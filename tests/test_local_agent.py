"""End-to-end over a real websocket, against the bundled reference agent.

Every other websocket test uses a fake socket. This one opens a real one on
localhost and drives a whole case through it, which is the only way to catch a
protocol mistake that both sides of a mock would agree on.
"""

import sys
from pathlib import Path

from websockets.asyncio.server import serve

from sayagain.adapters.websocket import WebSocketAdapter
from sayagain.expand import Case, expand
from sayagain.runner import Runner
from sayagain.scenario import load_scenario
from sayagain.tts import Synthesizer, ToneBackend

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "agents"))
import local_agent

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "reschedule_appointment.yaml"
HEARD = "I would like to reschedule my appointment to Friday morning."


def one_case() -> Case:
    scenario = load_scenario(EXAMPLE)
    return expand(
        scenario,
        languages=["en-US"],
        registers=["formal"],
        perturbations=["clean"],
        repeats=1,
    )[0]


async def test_a_case_runs_end_to_end_over_a_real_websocket(tmp_path: Path) -> None:
    async def handler(socket: object) -> None:
        session = local_agent.Session(socket, lambda pcm: HEARD, reply_ms=200)
        async for message in socket:  # type: ignore[attr-defined]
            await session.handle(message)

    async with serve(handler, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        runner = Runner(
            WebSocketAdapter(f"ws://localhost:{port}"),
            Synthesizer(ToneBackend(), cache_dir=tmp_path),
            realtime=False,
        )
        log = await runner.run(one_case())

    assert log.error is None, log.error
    kinds = [event.kind for event in log.events]
    assert "transcript" in kinds
    assert "audio" in kinds
    assert "end_turn" in kinds

    calls = [event.tool_call for event in log.events if event.kind == "tool_call"]
    assert calls, "the reference agent called no tool"
    assert calls[-1] is not None
    assert calls[-1].name == "reschedule_appointment"
    assert calls[-1].arguments == {"date": "friday", "time": "morning"}
    assert log.state == {"appointment.day": "friday", "appointment.time": "morning"}


async def test_the_reference_agent_reports_state_over_the_wire(tmp_path: Path) -> None:
    """`state` is a separate message, and it must reach the harness."""

    async def handler(socket: object) -> None:
        session = local_agent.Session(socket, lambda pcm: HEARD, reply_ms=100)
        async for message in socket:  # type: ignore[attr-defined]
            await session.handle(message)

    async with serve(handler, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        runner = Runner(
            WebSocketAdapter(f"ws://localhost:{port}"),
            Synthesizer(ToneBackend(), cache_dir=tmp_path),
            realtime=False,
        )
        log = await runner.run(one_case())

    assert log.state["appointment.day"] == "friday"


def test_the_reference_agent_needs_nothing_from_sayagain() -> None:
    """It is a reference for other languages, so it must not import the harness."""
    source = (
        Path(__file__).resolve().parent.parent / "examples" / "agents" / "local_agent.py"
    ).read_text()
    assert "import sayagain" not in source
    assert "from sayagain" not in source
