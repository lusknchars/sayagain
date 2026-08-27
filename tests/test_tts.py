"""Tests for speech synthesis and its cache. Nothing here touches the network."""

from pathlib import Path

import pytest

from sayagain.audio import SAMPLE_WIDTH, duration_ms
from sayagain.tts import BACKENDS, Synthesizer, ToneBackend, TTSError, Voice, get_backend


class CountingBackend:
    """Records how often it was actually asked to synthesise."""

    name = "counting"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    async def synthesize(self, text: str, *, language: str, voice: str | None = None) -> bytes:
        self.calls.append((text, language, voice))
        return b"\x00\x01" * 1600

    async def voices(self, language: str | None = None) -> list[Voice]:
        return [Voice(id="counting-1", language=language or "en-US", backend=self.name)]


async def test_the_second_identical_request_is_served_from_cache(tmp_path: Path) -> None:
    backend = CountingBackend()
    synth = Synthesizer(backend, cache_dir=tmp_path)

    first = await synth.say("Friday morning", language="en-US")
    second = await synth.say("Friday morning", language="en-US")

    assert first == second
    assert len(backend.calls) == 1


async def test_different_text_is_a_different_cache_entry(tmp_path: Path) -> None:
    backend = CountingBackend()
    synth = Synthesizer(backend, cache_dir=tmp_path)

    await synth.say("Friday morning", language="en-US")
    await synth.say("Thursday morning", language="en-US")

    assert len(backend.calls) == 2


async def test_the_voice_is_part_of_the_cache_key(tmp_path: Path) -> None:
    backend = CountingBackend()
    synth = Synthesizer(backend, cache_dir=tmp_path)

    await synth.say("Friday", language="en-US", voice="a")
    await synth.say("Friday", language="en-US", voice="b")

    assert len(backend.calls) == 2


async def test_the_backend_name_is_part_of_the_cache_key(tmp_path: Path) -> None:
    first = Synthesizer(CountingBackend(), cache_dir=tmp_path)
    second = Synthesizer(ToneBackend(), cache_dir=tmp_path)

    assert first.cache_path("Friday", language="en-US", voice=None) != second.cache_path(
        "Friday", language="en-US", voice=None
    )


async def test_the_cache_survives_a_new_synthesizer(tmp_path: Path) -> None:
    backend = CountingBackend()
    await Synthesizer(backend, cache_dir=tmp_path).say("Friday", language="en-US")

    await Synthesizer(backend, cache_dir=tmp_path).say("Friday", language="en-US")

    assert len(backend.calls) == 1


async def test_cached_audio_is_written_where_it_says(tmp_path: Path) -> None:
    synth = Synthesizer(CountingBackend(), cache_dir=tmp_path)

    await synth.say("Friday", language="en-US")

    assert synth.cache_path("Friday", language="en-US", voice=None).is_file()


# --- the offline backend --------------------------------------------------


async def test_the_tone_backend_needs_no_network_and_is_deterministic() -> None:
    backend = ToneBackend()

    first = await backend.synthesize("Friday morning", language="en-US")
    second = await backend.synthesize("Friday morning", language="en-US")

    assert first == second
    assert len(first) % SAMPLE_WIDTH == 0
    assert duration_ms(first) > 0


async def test_the_tone_backend_says_different_things_differently() -> None:
    backend = ToneBackend()

    assert await backend.synthesize("Friday", language="en-US") != await backend.synthesize(
        "Thursday", language="en-US"
    )


async def test_longer_text_takes_longer_to_say() -> None:
    backend = ToneBackend()

    short = await backend.synthesize("Friday", language="en-US")
    long = await backend.synthesize(
        "I would like to reschedule to Friday morning", language="en-US"
    )

    assert duration_ms(long) > duration_ms(short)


# --- the registry ---------------------------------------------------------


def test_the_registry_knows_the_bundled_backends() -> None:
    assert {"edge", "tone"} <= set(BACKENDS)


def test_get_backend_returns_something_that_can_speak() -> None:
    assert get_backend("tone").name == "tone"


def test_an_unknown_backend_names_itself() -> None:
    with pytest.raises(TTSError, match="elevenlabs"):
        get_backend("elevenlabs")
