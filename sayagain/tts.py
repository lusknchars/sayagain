"""Speech synthesis, behind a provider interface, with an on-disk cache.

Synthesis is the slowest and least reliable part of a run: `edge-tts` speaks to
an undocumented Microsoft endpoint that rate-limits by IP. So this module is
built as a registry rather than a hard dependency on one provider, and every
utterance is cached by content hash. A warm cache makes a run reproducible and
offline no matter which provider produced it.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

import numpy as np
import soundfile as sf
from scipy import signal

from sayagain.audio import SAMPLE_RATE, to_pcm16

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sayagain" / "tts"


class TTSError(Exception):
    """Synthesis could not be done."""


@dataclass(frozen=True, slots=True)
class Voice:
    """One voice a backend can speak with."""

    id: str
    language: str
    backend: str


class TTSBackend(Protocol):
    """Anything that can turn text into 16 kHz mono PCM16."""

    name: str

    async def synthesize(self, text: str, *, language: str, voice: str | None = None) -> bytes:
        """Speak `text` and return raw PCM16."""
        ...

    async def voices(self, language: str | None = None) -> list[Voice]:
        """List the voices this backend offers."""
        ...


class ToneBackend:
    """Offline, deterministic stand-in speech.

    It does not say words, so no ASR can understand it. That is the point: it
    lets the pipeline, the perturbations and the timing be exercised with no
    network and no model, which is what makes a dry run possible anywhere.
    """

    name = "tone"

    async def synthesize(self, text: str, *, language: str, voice: str | None = None) -> bytes:
        """Turn text into a deterministic warble whose length tracks the text."""
        digest = hashlib.sha256(f"{language}|{voice}|{text}".encode()).digest()
        base_hz = 140.0 + digest[0] / 255.0 * 120.0
        ms = max(400, min(8_000, 90 * len(text.split()) + 300))
        count = SAMPLE_RATE * ms // 1000
        t = np.arange(count) / SAMPLE_RATE
        wobble = np.sin(2 * np.pi * 4.5 * t) * 12.0
        samples = 0.4 * np.sin(2 * np.pi * (base_hz + wobble) * t)
        envelope = np.minimum(1.0, np.minimum(t * 20, (t[-1] - t) * 20))
        return to_pcm16(samples * envelope)

    async def voices(self, language: str | None = None) -> list[Voice]:
        """Return the single synthetic voice."""
        return [Voice(id="tone", language=language or "*", backend=self.name)]


class EdgeTTSBackend:
    """Microsoft Edge's online voices, via `edge-tts`.

    Free and the only bundled backend that covers every locale in the default
    matrix, `en-IN` included. It is also an unofficial endpoint that returns 403
    when it feels like it, which is why the cache matters.
    """

    name = "edge"

    #: One sensible default per locale in the documented matrix.
    DEFAULT_VOICES: ClassVar[dict[str, str]] = {
        "en-US": "en-US-JennyNeural",
        "en-IN": "en-IN-NeerjaNeural",
        "en-GB": "en-GB-SoniaNeural",
        "es-MX": "es-MX-DaliaNeural",
        "pt-BR": "pt-BR-FranciscaNeural",
        "hi-IN": "hi-IN-SwaraNeural",
        "de-DE": "de-DE-KatjaNeural",
    }

    async def synthesize(self, text: str, *, language: str, voice: str | None = None) -> bytes:
        """Speak `text`, converting the returned MP3 to PCM16."""
        try:
            import edge_tts
        except ImportError as error:  # pragma: no cover - depends on the install
            raise TTSError("edge-tts is not installed; pip install sayagain[all]") from error

        chosen = voice or self.DEFAULT_VOICES.get(language)
        if chosen is None:
            raise TTSError(
                f"no default edge voice for {language!r}; set matrix.voice for it "
                f"(run `sayagain voices --language {language}`)"
            )
        chunks = bytearray()
        try:
            async for chunk in edge_tts.Communicate(text, chosen).stream():
                if chunk["type"] == "audio":
                    chunks.extend(chunk["data"])
        except Exception as error:
            raise TTSError(f"edge-tts failed for {chosen}: {error}") from error
        if not chunks:
            raise TTSError(f"edge-tts returned no audio for {chosen}")
        return decode_to_pcm16(bytes(chunks))

    async def voices(self, language: str | None = None) -> list[Voice]:
        """Ask the service what it can speak with."""
        try:
            import edge_tts
        except ImportError as error:  # pragma: no cover - depends on the install
            raise TTSError("edge-tts is not installed; pip install sayagain[all]") from error

        found = await edge_tts.list_voices()
        return [
            Voice(id=entry["ShortName"], language=entry["Locale"], backend=self.name)
            for entry in found
            if language is None or entry["Locale"] == language
        ]


#: Registry of everything that can speak. Add your provider here.
BACKENDS: dict[str, type[TTSBackend]] = {
    "edge": EdgeTTSBackend,
    "tone": ToneBackend,
}


def get_backend(name: str) -> TTSBackend:
    """Build a backend by name.

    Raises:
        TTSError: no backend is registered under that name.

    """
    backend = BACKENDS.get(name)
    if backend is None:
        raise TTSError(f"unknown tts backend {name!r}; known backends are {sorted(BACKENDS)}")
    return backend()


class Synthesizer:
    """A backend plus a content-addressed cache of everything it has said."""

    def __init__(self, backend: TTSBackend, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.backend = backend
        self.cache_dir = cache_dir

    def cache_path(self, text: str, *, language: str, voice: str | None) -> Path:
        """Where this utterance lives on disk."""
        key = f"{self.backend.name}|{language}|{voice or 'default'}|{text}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.wav"

    async def say(self, text: str, *, language: str, voice: str | None = None) -> bytes:
        """Return PCM16 for `text`, synthesising only on a cache miss."""
        path = self.cache_path(text, language=language, voice=voice)
        if path.is_file():
            data, _ = sf.read(path, dtype="int16", always_2d=False)
            return bytes(np.asarray(data, dtype="<i2").tobytes())

        pcm = await self.backend.synthesize(text, language=language, voice=voice)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.frombuffer(pcm, dtype="<i2"), SAMPLE_RATE, subtype="PCM_16")
        return pcm


def decode_to_pcm16(data: bytes) -> bytes:
    """Decode any audio a backend returns into 16 kHz mono PCM16.

    Tries libsndfile first and falls back to ffmpeg, because MP3 support in
    libsndfile depends on how the wheel was built.
    """
    try:
        samples, rate = sf.read(io.BytesIO(data), dtype="float64", always_2d=False)
    except Exception:
        return _decode_with_ffmpeg(data)

    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = np.asarray(signal.resample_poly(audio, SAMPLE_RATE, rate), dtype=np.float64)
    return to_pcm16(audio)


def _decode_with_ffmpeg(data: bytes) -> bytes:
    if shutil.which("ffmpeg") is None:
        raise TTSError("cannot decode audio: libsndfile refused it and ffmpeg is not installed")
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise TTSError(f"ffmpeg could not decode the audio: {result.stderr.decode()[:200]}")
    return result.stdout
