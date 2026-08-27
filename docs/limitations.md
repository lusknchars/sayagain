# Limitations

Read this before quoting a number from sayagain anywhere that matters.

## TTS voices are not accented speakers

Every utterance is synthesised. An `en-IN` neural voice is a model trained to
sound like Indian English; it is not an Indian English speaker, and it does not
carry the prosody, disfluency, coarticulation or recording conditions of real
human speech.

This matters because the research motivating this tool measured *humans*.
Koenecke et al. (2020) found ~19 errors per 100 words for white speakers against
~35 for Black speakers across five commercial ASR systems; DiChristofano et al.
(2022) and Graham & Roll (2024) found the gap persists across global English
accents and widens on spontaneous rather than read speech. sayagain does not
reproduce that. It tests whether an agent survives *synthetic* accent and
register variation, which is a cheap proxy you can run in CI, not a measurement
of fairness across real speakers.

**A passing sayagain run is not evidence that your agent serves any group of
people well.** It is evidence that it survived a specific, synthetic matrix.

## One synthesis provider may flatter one recogniser

All bundled voices come from a single source. If that source and your ASR share
training data or acoustic conventions, your agent will score better than it
deserves, and you will not be able to see that from the report. Swapping the TTS
backend changes absolute numbers; only compare runs that used the same one.

## The noise beds are synthetic

`sayagain/assets/noise/cafe.wav` and `street.wav` are generated, not recorded. They
reproduce the spectral shape and modulation that degrade ASR, not the acoustics
of an actual room — no reverberation, no Lombard effect, no microphone. Point a
perturbation at your own recording when the answer matters:

```yaml
- id: noise
  params: { kind: /path/to/your/recording.wav, snr_db: 10 }
```

## Small n, and no confidence intervals

The default matrix is 126 cases against one agent. `repeats` gives you `pass@k`
and `pass^k`, and the gap between them separates flaky from broken — but neither
is a confidence interval, and none of the reported percentages come with one. Do
not read a difference of one or two cases as a real effect.

## Latency depends on your machine

`first_audio_ms` is measured on the machine running the harness, against an agent
that may be local. A cascade agent running whisper on CPU will miss a 1500 ms
budget on a laptop and meet it on a GPU box. Report the hardware alongside the
number, and treat latency as a comparison between runs rather than an absolute.

`--no-realtime` streams the user's audio as fast as the adapter accepts it
instead of at wall-clock speed. Processing time is still real, but the arrival
pattern is not, so anything sensitive to timing during the utterance — VAD
behaviour, mid-utterance barge-in — is not faithfully reproduced in that mode.

## The bundled agent is a toy

`--adapter mock` is a keyword matcher with an energy-based endpointer, there so
the demo and CI run without an API key. Findings about it — including that
background noise makes it *more* robust, by keeping its VAD engaged through
hesitation pauses — are facts about that endpointer, not about voice agents.
Point the harness at a real agent before believing anything.

## What is deliberately not here in v0.1

No LLM user simulator (turns are scripted), no LLM-as-judge scoring, no web
dashboard, no prosody or naturalness scoring, and no adapters for hosted
platforms. Each is an open issue rather than an oversight.
