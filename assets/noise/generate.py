"""Generate the synthetic noise beds shipped with sayagain.

These are not recordings. They are deterministic synthesis, so the repo carries
no third-party audio licence and `pip install sayagain` is self-contained. Point
`noise(kind=...)` at your own recordings when you need the real thing.

Run: uv run python assets/noise/generate.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

RATE = 16_000
SECONDS = 20
HERE = Path(__file__).parent


def _bandpass(data: np.ndarray, low: float, high: float) -> np.ndarray:
    sos = signal.butter(4, [low, high], btype="bandpass", fs=RATE, output="sos")
    return np.asarray(signal.sosfilt(sos, data), dtype=np.float64)


def cafe(rng: np.random.Generator) -> np.ndarray:
    """Babble: overlapping speech-band voices plus room tone."""
    count = RATE * SECONDS
    bed = np.zeros(count)
    for _ in range(12):
        voice = _bandpass(rng.standard_normal(count), 200.0, 3000.0)
        envelope = np.interp(
            np.arange(count),
            np.linspace(0, count, SECONDS * 6),
            rng.random(SECONDS * 6) ** 2,
        )
        bed += voice * envelope
    bed += _bandpass(rng.standard_normal(count), 50.0, 200.0) * 0.4  # room tone
    return bed


def street(rng: np.random.Generator) -> np.ndarray:
    """Traffic: brown-noise rumble with occasional passing swells."""
    count = RATE * SECONDS
    rumble = np.cumsum(rng.standard_normal(count))
    rumble = _bandpass(rumble, 40.0, 900.0)
    bed = rumble / (np.max(np.abs(rumble)) or 1.0)
    for _ in range(6):
        start = int(rng.integers(0, count - RATE * 2))
        length = RATE * 2
        swell = np.hanning(length) * _bandpass(rng.standard_normal(length), 300.0, 2500.0)
        bed[start : start + length] += swell * 0.5
    return bed


def main() -> None:
    """Write cafe.wav and street.wav next to this script."""
    for name, make in (("cafe", cafe), ("street", street)):
        bed = make(np.random.default_rng(20260826))
        peak = float(np.max(np.abs(bed))) or 1.0
        sf.write(HERE / f"{name}.wav", (bed / peak * 0.7), RATE, subtype="PCM_16")
        print(f"wrote {name}.wav")


if __name__ == "__main__":
    main()
