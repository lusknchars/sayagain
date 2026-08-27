"""Tests for the acoustic perturbation layer."""

import numpy as np
import pytest

from sayagain.audio import SAMPLE_RATE
from sayagain.perturb import (
    PRESETS,
    PerturbationError,
    apply,
    choppy,
    describe,
    gain,
    noise,
    pause,
    speed,
    telephone,
)
from sayagain.scenario import PerturbationSpec


def sine(seconds: float = 2.0, hz: float = 440.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return 0.5 * np.sin(2 * np.pi * hz * t)


def rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(data))))


def band_energy(data: np.ndarray, low: float, high: float) -> float:
    spectrum = np.abs(np.fft.rfft(data)) ** 2
    freqs = np.fft.rfftfreq(len(data), 1 / SAMPLE_RATE)
    return float(spectrum[(freqs >= low) & (freqs < high)].sum())


def dominant_hz(data: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(data))
    return float(np.fft.rfftfreq(len(data), 1 / SAMPLE_RATE)[int(np.argmax(spectrum))])


# --- noise ----------------------------------------------------------------


@pytest.mark.parametrize("snr_db", [20.0, 10.0, 5.0, 0.0])
def test_noise_hits_the_requested_snr(snr_db: float) -> None:
    clean = sine()

    mixed = noise(clean, kind="cafe", snr_db=snr_db)

    measured = 20 * np.log10(rms(clean) / rms(mixed - clean))
    assert abs(measured - snr_db) < 1.0


def test_noise_is_reproducible_for_a_seed() -> None:
    clean = sine()

    assert np.array_equal(
        noise(clean, kind="cafe", snr_db=10, seed=7), noise(clean, kind="cafe", snr_db=10, seed=7)
    )


def test_a_different_seed_uses_a_different_stretch_of_noise() -> None:
    clean = sine()

    assert not np.array_equal(
        noise(clean, kind="cafe", snr_db=10, seed=1), noise(clean, kind="cafe", snr_db=10, seed=2)
    )


def test_an_unknown_noise_bed_is_an_error() -> None:
    with pytest.raises(PerturbationError, match="airport"):
        noise(sine(), kind="airport", snr_db=10)


# --- telephone ------------------------------------------------------------


def test_telephone_removes_energy_outside_the_passband() -> None:
    wide = np.random.default_rng(0).standard_normal(SAMPLE_RATE * 2) * 0.1

    narrow = telephone(wide)

    assert band_energy(narrow, 4000, 8000) / band_energy(wide, 4000, 8000) < 0.01
    assert band_energy(narrow, 0, 200) / band_energy(wide, 0, 200) < 0.1


def test_telephone_keeps_the_speech_band() -> None:
    wide = np.random.default_rng(0).standard_normal(SAMPLE_RATE * 2) * 0.1

    narrow = telephone(wide)

    assert band_energy(narrow, 500, 3000) / band_energy(wide, 500, 3000) > 0.3


def test_telephone_preserves_length() -> None:
    assert len(telephone(sine())) == len(sine())


# --- speed ----------------------------------------------------------------


def test_speeding_up_shortens_the_audio() -> None:
    original = sine()

    faster = speed(original, factor=1.2)

    assert abs(len(faster) - len(original) / 1.2) / len(original) < 0.02


def test_speeding_up_preserves_pitch() -> None:
    original = sine(hz=440.0)

    faster = speed(original, factor=1.2)

    assert abs(dominant_hz(faster) - 440.0) < 15.0


def test_speed_of_one_changes_nothing() -> None:
    original = sine()

    assert np.array_equal(speed(original, factor=1.0), original)


# --- gain, choppy, pause --------------------------------------------------


def test_gain_of_minus_six_db_halves_the_level() -> None:
    quieter = gain(sine(), db=-6.0)

    assert abs(rms(quieter) / rms(sine()) - 0.501) < 0.01


def test_choppy_drops_about_the_requested_share() -> None:
    steady = np.ones(SAMPLE_RATE * 4)

    chopped = choppy(steady, drop_ratio=0.1, burst_ms=60)

    assert abs(float(np.mean(chopped == 0.0)) - 0.1) < 0.03


def test_choppy_is_reproducible_for_a_seed() -> None:
    steady = np.ones(SAMPLE_RATE * 4)

    assert np.array_equal(
        choppy(steady, drop_ratio=0.1, seed=3), choppy(steady, drop_ratio=0.1, seed=3)
    )


def test_pause_inserts_silence_and_lengthens_the_audio() -> None:
    original = sine(seconds=1.0)

    paused = pause(original, at_ms=500, duration_ms=300)

    assert len(paused) == len(original) + SAMPLE_RATE * 300 // 1000
    inserted = paused[SAMPLE_RATE * 500 // 1000 : SAMPLE_RATE * 800 // 1000]
    assert np.all(inserted == 0.0)


# --- the registry the runner uses -----------------------------------------


def test_every_preset_the_example_scenario_uses_exists() -> None:
    assert {"clean", "telephone", "cafe_10db", "street_5db", "fast", "choppy"} <= set(PRESETS)


def test_clean_is_the_identity() -> None:
    original = sine()

    assert np.array_equal(apply(original, PerturbationSpec(id="clean"), seed=1), original)


def test_apply_dispatches_to_the_preset() -> None:
    original = sine()

    assert len(apply(original, PerturbationSpec(id="fast"), seed=1)) < len(original)


def test_inline_params_override_the_preset() -> None:
    original = sine()

    loud = apply(original, PerturbationSpec(id="cafe_10db", params={"snr_db": 0.0}), seed=1)

    measured = 20 * np.log10(rms(original) / rms(loud - original))
    assert abs(measured) < 1.0


def test_an_unknown_preset_names_itself() -> None:
    with pytest.raises(PerturbationError, match="underwater"):
        apply(sine(), PerturbationSpec(id="underwater"), seed=1)


def test_describe_records_the_parameters_for_the_report() -> None:
    assert describe(PerturbationSpec(id="cafe_10db")) == {
        "id": "cafe_10db",
        "kind": "cafe",
        "snr_db": 10.0,
    }
