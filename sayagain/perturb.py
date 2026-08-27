"""Seeded, composable acoustic perturbations.

Every function takes and returns float samples at 16 kHz and does one thing.
They are deliberately pure so a case can be reproduced exactly from its id and
seed, and so the report can say which knob broke the agent.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy import signal

from sayagain.audio import SAMPLE_RATE
from sayagain.scenario import PerturbationSpec

Samples = NDArray[np.float64]

NOISE_DIR = Path(__file__).resolve().parent.parent / "assets" / "noise"

TELEPHONE_BAND_HZ = (300.0, 3400.0)
TELEPHONE_RATE = 8_000


class PerturbationError(Exception):
    """A perturbation was asked for something it cannot do."""


def gain(samples: Samples, *, db: float, seed: int = 0) -> Samples:
    """Scale the level by `db` decibels. `seed` is unused; every knob has one."""
    return np.asarray(samples * (10.0 ** (db / 20.0)), dtype=np.float64)


def noise(samples: Samples, *, kind: str, snr_db: float, seed: int = 0) -> Samples:
    """Mix a background bed in at the requested signal-to-noise ratio.

    The gain is computed from the measured RMS of both signals, so the SNR is
    what was asked for rather than whatever the file happened to be recorded at.
    """
    bed = _fit(_load_bed(kind), len(samples), seed=seed)
    speech_rms = _rms(samples)
    bed_rms = _rms(bed)
    if speech_rms == 0.0 or bed_rms == 0.0:
        return samples.copy()
    wanted_bed_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    return np.asarray(samples + bed * (wanted_bed_rms / bed_rms), dtype=np.float64)


def telephone(samples: Samples, *, seed: int = 0) -> Samples:
    """Band-limit to 300-3400 Hz and round-trip through 8 kHz."""
    low, high = TELEPHONE_BAND_HZ
    sos = signal.butter(6, [low, high], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    filtered = signal.sosfilt(sos, samples)
    down = signal.resample_poly(filtered, TELEPHONE_RATE, SAMPLE_RATE)
    up = signal.resample_poly(down, SAMPLE_RATE, TELEPHONE_RATE)
    return _match_length(np.asarray(up, dtype=np.float64), len(samples))


def speed(samples: Samples, *, factor: float, seed: int = 0) -> Samples:
    """Time-stretch by `factor` with pitch preserved (WSOLA).

    Written out rather than pulled from librosa or rubberband: rubberband needs
    a system binary and librosa is a heavy dependency for one knob.
    """
    if math.isclose(factor, 1.0):
        return samples.copy()
    if factor <= 0:
        raise PerturbationError(f"speed factor must be positive, got {factor}")

    frame = 1024
    hop_out = frame // 2
    tolerance = 256
    window = np.hanning(frame)
    total_out = round(len(samples) / factor)

    accumulated = np.zeros(total_out + frame)
    weights = np.zeros(total_out + frame)
    previous_tail = np.zeros(hop_out)
    out_position = 0
    step = 0

    while out_position + frame <= len(accumulated):
        expected = round(step * hop_out * factor)
        lowest = max(0, expected - tolerance)
        highest = min(len(samples) - frame - hop_out, expected + tolerance)
        if highest <= lowest:
            break
        best = _best_offset(samples, previous_tail, lowest, highest, hop_out)
        accumulated[out_position : out_position + frame] += samples[best : best + frame] * window
        weights[out_position : out_position + frame] += window
        previous_tail = samples[best + hop_out : best + hop_out + hop_out]
        out_position += hop_out
        step += 1

    stretched = accumulated[:total_out] / np.maximum(weights[:total_out], 1e-6)
    return np.asarray(stretched, dtype=np.float64)


def choppy(samples: Samples, *, drop_ratio: float, burst_ms: int = 60, seed: int = 0) -> Samples:
    """Zero out bursts of samples, the way lost packets sound."""
    if not 0.0 <= drop_ratio < 1.0:
        raise PerturbationError(f"drop_ratio must be in [0, 1), got {drop_ratio}")
    chopped = samples.copy()
    burst = max(1, SAMPLE_RATE * burst_ms // 1000)
    wanted = int(len(samples) * drop_ratio)
    if wanted == 0:
        return chopped

    rng = np.random.default_rng(seed)
    dropped = 0
    guard = 0
    while dropped < wanted and guard < 10_000:
        guard += 1
        start = int(rng.integers(0, max(1, len(samples) - burst)))
        window = chopped[start : start + burst]
        dropped += int(np.count_nonzero(window))
        chopped[start : start + burst] = 0.0
    return chopped


def pause(samples: Samples, *, at_ms: int, duration_ms: int, seed: int = 0) -> Samples:
    """Insert silence mid-utterance, the way a speaker hesitates."""
    cut = min(len(samples), SAMPLE_RATE * at_ms // 1000)
    inserted = np.zeros(SAMPLE_RATE * duration_ms // 1000)
    return np.concatenate([samples[:cut], inserted, samples[cut:]])


def identity(samples: Samples, *, seed: int = 0) -> Samples:
    """Leave the audio alone. The control condition."""
    return samples


@dataclass(frozen=True, slots=True)
class Preset:
    """A named perturbation plus the parameters that define it."""

    func: Callable[..., Samples]
    params: dict[str, Any] = field(default_factory=dict)


#: The ids a scenario can name. Inline `params` override these.
PRESETS: dict[str, Preset] = {
    "clean": Preset(identity),
    "telephone": Preset(telephone),
    "cafe_10db": Preset(noise, {"kind": "cafe", "snr_db": 10.0}),
    "street_5db": Preset(noise, {"kind": "street", "snr_db": 5.0}),
    "cafe_0db": Preset(noise, {"kind": "cafe", "snr_db": 0.0}),
    "noise": Preset(noise, {"kind": "cafe", "snr_db": 10.0}),
    "fast": Preset(speed, {"factor": 1.2}),
    "slow": Preset(speed, {"factor": 0.8}),
    "speed": Preset(speed, {"factor": 1.2}),
    "choppy": Preset(choppy, {"drop_ratio": 0.1, "burst_ms": 60}),
    "quiet": Preset(gain, {"db": -12.0}),
    "gain": Preset(gain, {"db": -12.0}),
    "pause": Preset(pause, {"at_ms": 600, "duration_ms": 400}),
}


def apply(samples: Samples, spec: PerturbationSpec, *, seed: int = 42) -> Samples:
    """Run one perturbation over the audio, honouring inline parameter overrides."""
    preset = PRESETS.get(spec.id)
    if preset is None:
        raise PerturbationError(
            f"unknown perturbation {spec.id!r}; known ids are {sorted(PRESETS)}"
        )
    return preset.func(samples, seed=seed, **{**preset.params, **spec.params})


def describe(spec: PerturbationSpec) -> dict[str, Any]:
    """Return the parameters a case actually ran with, for the report."""
    preset = PRESETS.get(spec.id)
    base = dict(preset.params) if preset else {}
    return {"id": spec.id, **base, **spec.params}


def _best_offset(
    samples: Samples, previous_tail: Samples, lowest: int, highest: int, hop: int
) -> int:
    """Pick the input offset whose start best continues the last output frame."""
    best_offset = lowest
    best_score = -math.inf
    for candidate in range(lowest, highest + 1, 8):
        score = float(np.dot(samples[candidate : candidate + hop], previous_tail))
        if score > best_score:
            best_score = score
            best_offset = candidate
    return best_offset


@lru_cache(maxsize=8)
def _load_bed(kind: str) -> Samples:
    path = Path(kind) if Path(kind).suffix else NOISE_DIR / f"{kind}.wav"
    if not path.is_file():
        raise PerturbationError(
            f"no noise bed {kind!r}; expected {path} (bundled beds: cafe, street)"
        )
    data, rate = sf.read(path, dtype="float64", always_2d=False)
    bed = np.asarray(data, dtype=np.float64)
    if bed.ndim > 1:
        bed = bed.mean(axis=1)
    if rate != SAMPLE_RATE:
        bed = np.asarray(signal.resample_poly(bed, SAMPLE_RATE, rate), dtype=np.float64)
    return bed


def _fit(bed: Samples, length: int, *, seed: int) -> Samples:
    """Take `length` samples from the bed, starting at a seeded offset."""
    if len(bed) < length:
        bed = np.tile(bed, int(np.ceil(length / len(bed))))
    offset = int(np.random.default_rng(seed).integers(0, max(1, len(bed) - length)))
    return bed[offset : offset + length]


def _match_length(data: Samples, length: int) -> Samples:
    if len(data) >= length:
        return data[:length]
    return np.concatenate([data, np.zeros(length - len(data))])


def _rms(data: Samples) -> float:
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data))))
