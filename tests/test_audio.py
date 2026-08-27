"""Tests for the 16 kHz mono PCM16 audio helpers."""

import math

from sayagain.audio import (
    FRAME_BYTES,
    FRAME_MS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    duration_ms,
    frames,
    has_speech,
    resample_pcm16,
    rms,
    silence,
    to_float,
    to_pcm16,
    tone,
    trim_silence,
)


def test_frame_size_is_20ms_of_16khz_pcm16() -> None:
    assert SAMPLE_RATE == 16_000
    assert FRAME_MS == 20
    assert SAMPLE_WIDTH == 2
    assert FRAME_BYTES == SAMPLE_RATE * FRAME_MS // 1000 * SAMPLE_WIDTH == 640


def test_one_second_of_silence_is_fifty_frames() -> None:
    chunks = list(frames(silence(1000)))

    assert len(chunks) == 50
    assert {len(chunk) for chunk in chunks} == {FRAME_BYTES}


def test_a_ragged_tail_is_padded_to_a_whole_frame() -> None:
    chunks = list(frames(silence(20) + b"\x00\x00"))

    assert len(chunks) == 2
    assert len(chunks[1]) == FRAME_BYTES


def test_silence_carries_no_speech() -> None:
    frame = next(iter(frames(silence(20))))

    assert rms(frame) == 0.0
    assert has_speech(frame) is False


def test_a_tone_carries_speech() -> None:
    frame = next(iter(frames(tone(100))))

    assert rms(frame) > 0.1
    assert has_speech(frame) is True


def test_tone_is_deterministic() -> None:
    assert tone(100) == tone(100)


def test_pcm_and_float_round_trip() -> None:
    original = tone(100)

    samples = to_float(original)

    assert samples.max() <= 1.0
    assert samples.min() >= -1.0
    assert math.isclose(rms(to_pcm16(samples)), rms(original), abs_tol=1e-4)


def test_resampling_changes_length_by_the_rate_ratio() -> None:
    original = tone(1000)

    upsampled = resample_pcm16(original, from_rate=16_000, to_rate=24_000)

    assert abs(len(upsampled) / len(original) - 1.5) < 0.01


def test_resampling_round_trip_preserves_duration() -> None:
    original = tone(500)

    there = resample_pcm16(original, from_rate=16_000, to_rate=24_000)
    back = resample_pcm16(there, from_rate=24_000, to_rate=16_000)

    assert abs(duration_ms(back) - duration_ms(original)) <= 1


def test_resampling_to_the_same_rate_is_a_no_op() -> None:
    original = tone(100)

    assert resample_pcm16(original, from_rate=16_000, to_rate=16_000) == original


def test_trimming_removes_leading_and_trailing_silence() -> None:
    padded = silence(500) + tone(1000) + silence(500)

    trimmed = trim_silence(padded, keep_ms=40)

    assert duration_ms(trimmed) < duration_ms(padded)
    assert abs(duration_ms(trimmed) - 1080) < 60  # the tone plus 40 ms either side


def test_trimming_keeps_audio_that_is_already_tight() -> None:
    tight = tone(500)

    assert duration_ms(trim_silence(tight, keep_ms=40)) == duration_ms(tight)


def test_trimming_pure_silence_changes_nothing() -> None:
    quiet = silence(500)

    assert trim_silence(quiet) == quiet


def test_trimming_leaves_internal_pauses_alone() -> None:
    # A pause inside an utterance is signal, not padding: it is exactly what
    # trips an endpointer, and removing it would hide the failure.
    utterance = tone(300) + silence(600) + tone(300)

    assert duration_ms(trim_silence(utterance, keep_ms=40)) == duration_ms(utterance)
