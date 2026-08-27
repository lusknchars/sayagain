"""16 kHz mono PCM16 helpers.

One format crosses every boundary in sayagain: 16 kHz, mono, signed 16-bit
little-endian, in 20 ms frames of 640 bytes. Adapters convert to and from this
at their own edge so nothing upstream has to care what an agent speaks natively.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000
FRAME_MS = 20
SAMPLE_WIDTH = 2
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = SAMPLES_PER_FRAME * SAMPLE_WIDTH

#: Frames quieter than this count as silence when detecting speech.
SPEECH_RMS_THRESHOLD = 0.02

_FULL_SCALE = 32768.0


def frames(pcm: bytes) -> Iterator[bytes]:
    """Split PCM16 bytes into 20 ms frames, zero-padding a ragged tail."""
    for start in range(0, len(pcm), FRAME_BYTES):
        frame = pcm[start : start + FRAME_BYTES]
        if len(frame) < FRAME_BYTES:
            frame = frame + b"\x00" * (FRAME_BYTES - len(frame))
        yield frame


def silence(ms: int) -> bytes:
    """Return `ms` milliseconds of digital silence."""
    return b"\x00" * (SAMPLE_RATE * ms // 1000 * SAMPLE_WIDTH)


def tone(ms: int, hz: float = 220.0, amplitude: float = 0.3) -> bytes:
    """Return a deterministic sine tone, used as stand-in agent speech."""
    count = SAMPLE_RATE * ms // 1000
    step = 2 * math.pi * hz / SAMPLE_RATE
    samples = amplitude * np.sin(step * np.arange(count, dtype=np.float64))
    return to_pcm16(samples)


def to_float(pcm: bytes) -> NDArray[np.float64]:
    """Decode PCM16 bytes into floats in [-1, 1]."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64) / _FULL_SCALE


def to_pcm16(samples: NDArray[np.float64]) -> bytes:
    """Encode floats in [-1, 1] as PCM16 bytes, clipping anything louder."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * (_FULL_SCALE - 1)).astype("<i2").tobytes()


def rms(pcm: bytes) -> float:
    """Root-mean-square level of a frame, on the same [0, 1] scale as `to_float`."""
    samples = to_float(pcm)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def has_speech(pcm: bytes, threshold: float = SPEECH_RMS_THRESHOLD) -> bool:
    """Report whether a frame is loud enough to count as speech."""
    return rms(pcm) > threshold


def trim_silence(
    pcm: bytes, *, threshold: float = SPEECH_RMS_THRESHOLD, keep_ms: int = 40
) -> bytes:
    """Trim silence from the ends of an utterance, keeping `keep_ms` of padding.

    Synthesised speech comes with silence bolted on either end, which is not
    part of what the caller said. Left in, it trips the agent's endpointer
    before the audio has finished arriving, and `first_audio_ms` then measures
    from the wrong instant. Pauses *inside* the utterance are left alone: those
    are the thing under test.
    """
    chunks = list(frames(pcm))
    loud = [index for index, frame in enumerate(chunks) if has_speech(frame, threshold)]
    if not loud:
        return pcm
    pad = max(0, keep_ms // FRAME_MS)
    start = max(0, loud[0] - pad)
    stop = min(len(chunks), loud[-1] + 1 + pad)
    return b"".join(chunks[start:stop])


def resample_pcm16(pcm: bytes, *, from_rate: int, to_rate: int) -> bytes:
    """Convert PCM16 between sample rates.

    Adapters call this at their own edge: a backend that speaks 24 kHz is its
    own problem, and nothing above the adapter should ever see another rate.
    """
    if from_rate == to_rate:
        return pcm
    from scipy import signal

    samples = to_float(pcm)
    divisor = math.gcd(from_rate, to_rate)
    converted = signal.resample_poly(samples, to_rate // divisor, from_rate // divisor)
    return to_pcm16(np.asarray(converted, dtype=np.float64))


def duration_ms(pcm: bytes) -> int:
    """Duration of PCM16 bytes in whole milliseconds."""
    return len(pcm) // SAMPLE_WIDTH * 1000 // SAMPLE_RATE
