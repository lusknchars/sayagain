# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-08-27

### Added
- Scenario schema (`sayagain/scenario.py`) with YAML loading and validation.
- 16 kHz mono PCM16 audio helpers (`sayagain/audio.py`).
- The `Adapter` / `AgentSession` contract (`sayagain/adapters/base.py`).
- In-process mock agent (`sayagain/adapters/mock.py`) with an injectable
  transcriber, so tests need no ASR model.
- Matrix expansion (`sayagain/expand.py`): register x language x perturbation x repeat.
- `sayagain run`, `sayagain new`, `sayagain voices`, `sayagain doctor`.
- Three example scenarios in six languages: `reschedule_appointment`,
  `order_status` (digits through a channel) and `transfer_money` (a date whose
  meaning changes with the caller's locale).
- Spoken month-and-day dates in all six languages, because text-to-speech reads
  `09/04` aloud in its own locale: an agent hears "September 4th" or "9. April",
  never the digits.
- Cross-language argument normalisation (`sayagain/normalize.py`): weekdays and
  parts of day in all six matrix languages, locale-dependent numeric dates, and
  the `mañana`/`morgen` tomorrow-vs-morning ambiguity resolved by field.
- Perturbations (`sayagain/perturb.py`): noise at a measured SNR, telephone
  band-limiting, WSOLA time-stretch, gain, packet loss, mid-utterance pauses.
- Speech synthesis behind a provider registry with a content-addressed cache
  (`sayagain/tts.py`): `edge` for real voices, `tone` for offline dry runs.
- Case runner (`sayagain/runner.py`) with real barge-in and JSONL event logs.
- Scoring (`sayagain/score.py`): first-audio and barge-in latency, tool-call and
  end-state correctness, transcript WER, `failure_locus` attribution, and
  aggregation with pass rate, pass@k, pass^k and latency percentiles.
- Generic WebSocket adapter plus `docs/adapter-protocol.md`, so an agent in any
  language can be tested.
- OpenAI Realtime adapter, with 24 kHz conversion at the adapter boundary.
- `--end-silence-ms` and `--mock-endpoint-ms` knobs: endpointing thresholds move
  results, so they must not be buried constants.

- Report artifacts (`sayagain/report.py`): terminal tables, `report.json`,
  `report.md` and `heatmap.png`, all reported **per assertion** rather than as a
  single pass rate, and grouped by register as well as by language and
  perturbation.
- `doctor` now reports the whisper model, cached utterances, noise beds, and
  which adapters and voice backends are available.

### Fixed
- Barge-in was scored against the agent's *answer to the correction*, so every
  well-behaved agent failed `max_barge_in_stop_ms`. It is now measured only when
  the agent was actually speaking as the interruption landed.
- The mock transcribed inline, stalling the runner's audio stream and corrupting
  every timestamp after it. It now answers off to the side, as a separate
  process would.
- Synthesised speech carries silence at both ends, which ended the agent's turn
  before the audio had finished arriving. The runner trims it before perturbing;
  pauses *inside* an utterance are left alone, being the thing under test.

### Notes
- The bundled noise beds are synthesised, not recorded. See
  `sayagain/assets/noise/LICENSE.md`.
- `AgentSession.events()` is declared non-async in the protocol; async generator
  functions return an `AsyncIterator` rather than awaiting one, so declaring it
  `async def` would make every real implementation fail type checking.

[0.1.0]: https://github.com/lusknchars/sayagain/releases/tag/v0.1.0
