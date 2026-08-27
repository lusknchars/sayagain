# Adapter protocol

There are two ways to make an agent testable with sayagain. Write a Python
adapter, or expose a WebSocket that speaks the protocol below. The second one
works from any language and takes about an hour.

## The audio format, once

Everything is **16 kHz, mono, signed 16-bit little-endian PCM**, in **20 ms
frames of exactly 640 bytes**. Both directions. No headers, no container.

If your agent speaks something else — the OpenAI Realtime API uses 24 kHz, most
telephony is 8 kHz — convert at your edge. Nothing above the adapter is allowed
to see another format, which is what keeps latency numbers comparable between
agents.

## WebSocket protocol

Connect: sayagain opens a client connection to the `--url` you pass. Your agent
is the server.

### What sayagain sends

| Message | Meaning |
|---|---|
| binary frame | 640 bytes of user audio |
| `{"type": "start", "system_prompt": str \| null, "tools": [...], "sample_rate": 16000, "frame_ms": 20}` | first message on every session |
| `{"type": "silence", "ms": 600}` | the user stopped talking for this long |

`silence` is sent instead of shipping zeros, so you can drive your own endpointer
from it without paying for the bytes. If your VAD needs real samples, treat it as
`ms` worth of zeros.

`tools` is the scenario's tool list verbatim, in sayagain's compact form:

```json
[{"name": "reschedule_appointment", "schema": {"date": "string", "time": "string"}}]
```

### What your agent sends

| Message | Meaning |
|---|---|
| binary frame | 640 bytes of agent audio |
| `{"type": "transcript", "text": "..."}` | what you heard the user say |
| `{"type": "tool_call", "name": "...", "arguments": {...}}` | a tool you decided to call |
| `{"type": "end_turn"}` | you are done speaking |
| `{"type": "state", "state": {"appointment.day": "friday"}}` | your end state, dotted paths |
| `{"type": "error", "message": "..."}` | something went wrong |

Anything else is ignored, so you may send your own bookkeeping over the same
socket.

### Two messages that are worth the effort

**`transcript`.** Without it, every failure is attributed as `unknown`. With it,
sayagain can tell you whether a case failed because the words never arrived
(`asr`) or because they arrived and were misused (`reasoning`). That single
distinction is most of the value of running this at all.

**Stopping when talked over.** Barge-in is measured as the time from the first
frame of interrupting user audio to your last frame of outgoing audio. An agent
that queues its whole reply up front and ignores incoming audio will score
badly, and should.

### A minimal agent

```python
import asyncio, json, websockets


async def agent(ws):
    async for message in ws:
        if isinstance(message, bytes):
            continue  # buffer user audio
        event = json.loads(message)
        if event["type"] == "silence" and event["ms"] >= 400:
            await ws.send(json.dumps({"type": "transcript", "text": "..."}))
            await ws.send(
                json.dumps(
                    {
                        "type": "tool_call",
                        "name": "reschedule_appointment",
                        "arguments": {"date": "friday", "time": "morning"},
                    }
                )
            )
            for frame in reply_frames():  # 640 bytes each
                await ws.send(frame)
            await ws.send(json.dumps({"type": "state", "state": {"appointment.day": "friday"}}))
            await ws.send(json.dumps({"type": "end_turn"}))


asyncio.run(websockets.serve(agent, "localhost", 8765).wait_closed())
```

Then:

```bash
sayagain run examples/ --adapter websocket --url ws://localhost:8765
```

## Python adapters

Implement two protocols from `sayagain/adapters/base.py`:

```python
class Adapter(Protocol):
    name: str

    async def session(self, *, system_prompt: str | None, tools: list[dict]) -> AgentSession: ...


class AgentSession(Protocol):
    async def send_audio(self, frame: bytes) -> None: ...
    async def send_silence(self, ms: int) -> None: ...
    def events(self) -> AsyncIterator[AgentEvent]: ...
    async def state(self) -> dict: ...
    async def close(self) -> None: ...
```

`events()` is not declared `async def`: an async generator function *returns* an
`AsyncIterator` rather than awaiting one, so declaring it `async def` would make
every real implementation fail type checking.

Every event carries `t_ns`, nanoseconds since the session started
(`sayagain.adapters.base.Clock` does this for you). `sayagain/adapters/mock.py`
is about 200 readable lines and is the reference implementation.
