# Writing scenarios

A scenario holds one intent constant and varies how it is said. The schema lives
in `sayagain/scenario.py`; `examples/reschedule_appointment.yaml` is the worked
example. Everything below is validated at load time, so a typo is an error
rather than a case that silently never runs.

## The shape

```yaml
id: reschedule_appointment          # [a-z0-9_-], used in report rows and log filenames
description: User wants to move an existing appointment to Friday morning.
agent:
  system_prompt_file: prompts/clinic.md   # resolved relative to this file
  # or: system_prompt: "You are ..."      # one or the other, not both
  tools:
    - name: reschedule_appointment
      schema: { date: string, time: string }
matrix:
  language: [en-US, pt-BR]
  voice: default                    # or { pt-BR: pt-BR-FranciscaNeural }
  perturbation: [clean, telephone]  # or the long form, below
repeats: 3
turns:
  - user:
      intent: reschedule_to_friday_morning
      variants:
        formal:   { en-US: "...", pt-BR: "..." }
        casual:   { en-US: "...", pt-BR: "..." }
    interrupt:
      after_agent_speaks_ms: 800
      with: { en-US: "Sorry — Friday, not Thursday." }
    expect:
      tool_call:
        name: reschedule_appointment
        arguments: { date: "friday", time: "morning" }
      max_first_audio_ms: 1500
      max_barge_in_stop_ms: 500
      end_state: { appointment.day: "friday" }
```

## How the matrix expands

`register x language x perturbation x repeat`. The example above is 2 registers
x 2 languages x 2 perturbations x 3 repeats = 24 cases.

**A register with no line in a language is skipped, not failed.** Partial
translation is the normal state of a scenario you are growing, so `codeswitch`
existing only in `en-US` costs you nothing. A variant language the matrix never
asks for is kept but reported by `sayagain doctor`, because it is usually a typo.

## Registers

The register names are yours; nothing in the code knows what `formal` means. The
four in the example are the ones worth having:

| register | what it tests |
|---|---|
| `formal` | the sentence your prompt was written for |
| `casual` | contractions, vague nouns, shorter |
| `disfluent` | fillers, restarts, mid-sentence pauses |
| `codeswitch` | a word or two from another language |

`disfluent` is the one that finds bugs. Internal pauses trip endpointers, and an
agent that answers half an utterance fails without any noise at all.

## Perturbations

Short form is an id; long form adds parameters.

```yaml
perturbation:
  - clean
  - telephone
  - { id: noise, params: { kind: cafe, snr_db: 0 } }
  - { id: speed, params: { factor: 1.4 } }
```

Bundled ids: `clean`, `telephone`, `cafe_10db`, `cafe_0db`, `street_5db`,
`noise`, `fast`, `slow`, `speed`, `choppy`, `quiet`, `gain`, `pause`.
`kind` also accepts a path to your own recording.

## Expectations

At least one is required, or the case could never fail.

- `tool_call` — name must be declared in `agent.tools`. Arguments are compared
  through the normalizer, so `friday` matches `sexta-feira`, `viernes` and
  `Freitag`. Numeric dates are read per locale: `09/04` is 4 September in
  `en-US` and 9 April everywhere else.
- `end_state` — dotted paths against whatever the agent reports as its state.
- `max_first_audio_ms` — end of user speech to first agent audio.
- `max_barge_in_stop_ms` — first frame of interrupting audio to the agent's last
  frame. An agent that was not speaking cannot talk over you, so that is not
  counted as a failure.

## Repeats, and what pass@k means

Voice agents are not deterministic. `repeats: 3` runs every case three times and
the report gives you both readings: `pass@k` is the share of cases that passed at
least once, `pass^k` the share that passed every time. A wide gap between them
means flaky, not broken — and it is the gap, not either number alone, that tells
you which.

## Knobs that move results

`--end-silence-ms` (how much silence sayagain sends to end a turn) and
`--mock-endpoint-ms` (when the bundled toy agent decides you stopped) change
outcomes materially, especially for `disfluent`. They are flags rather than
buried constants for that reason. Report the values you used alongside any
numbers you publish.
