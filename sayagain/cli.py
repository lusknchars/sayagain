"""The `sayagain` command line."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from sayagain import __version__
from sayagain.adapters.base import Adapter
from sayagain.adapters.mock import MockAdapter
from sayagain.adapters.openai_realtime import OpenAIRealtimeAdapter
from sayagain.adapters.websocket import WebSocketAdapter
from sayagain.expand import Case, expand
from sayagain.report import render_table, write_report
from sayagain.runner import RunLog, Runner
from sayagain.scenario import ScenarioError, load_scenarios
from sayagain.score import CaseResult, aggregate, score
from sayagain.tts import BACKENDS, Synthesizer, TTSError, get_backend

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

ADAPTERS = ("mock", "websocket", "openai_realtime")

STARTER = """\
id: {id}
description: "TODO: one sentence on what the caller wants."
agent:
  system_prompt: You are a helpful assistant. Call the tool when the caller asks.
  tools:
    - name: {id}
      schema: {{ date: string, time: string }}
matrix:
  language: [en-US, pt-BR]
  voice: default
  perturbation:
    - id: clean
    - id: telephone
    - id: cafe_10db
repeats: 1
turns:
  - user:
      intent: TODO_intent_name
      variants:
        formal:
          en-US: "TODO: how a careful speaker would say it."
          pt-BR: "TODO: como uma pessoa formal diria."
        casual:
          en-US: "TODO: how someone in a hurry would say it."
    expect:
      tool_call:
        name: {id}
        arguments: {{ date: "friday", time: "morning" }}
      max_first_audio_ms: 1500
"""


@app.command()
def run(
    target: Annotated[Path, typer.Argument(help="A scenario file or a directory of them.")],
    adapter: Annotated[str, typer.Option(help="Which agent backend to test.")] = "mock",
    url: Annotated[str | None, typer.Option(help="WebSocket URL for --adapter websocket.")] = None,
    tts: Annotated[str, typer.Option(help="Which voice backend speaks the user.")] = "edge",
    repeats: Annotated[int | None, typer.Option(help="Override the scenario's repeats.")] = None,
    only_language: Annotated[list[str] | None, typer.Option(help="Restrict languages.")] = None,
    only_perturbation: Annotated[
        list[str] | None, typer.Option(help="Restrict perturbations.")
    ] = None,
    only_register: Annotated[
        list[str] | None, typer.Option(help="Restrict registers, e.g. disfluent.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed for every perturbation.")] = 42,
    realtime: Annotated[
        bool, typer.Option(help="Stream audio at wall-clock speed, as a real caller would.")
    ] = True,
    end_silence_ms: Annotated[
        int, typer.Option(help="Silence sent after each utterance to end the turn.")
    ] = 600,
    mock_endpoint_ms: Annotated[
        int, typer.Option(help="How much silence the mock agent treats as end of turn.")
    ] = 400,
    green: Annotated[float, typer.Option(help="Pass rate at or above this shows green.")] = 0.9,
    yellow: Annotated[float, typer.Option(help="Pass rate at or above this shows yellow.")] = 0.7,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the matrix without running anything.")
    ] = False,
    out: Annotated[Path, typer.Option(help="Where reports are written.")] = Path("sayagain-report"),
) -> None:
    """Run scenarios against an agent and report what survived."""
    if adapter not in ADAPTERS:
        console.print(f"[red]unknown adapter {adapter!r}[/red]; expected one of {list(ADAPTERS)}")
        raise typer.Exit(2)
    if adapter == "websocket" and not url and not dry_run:
        console.print("[red]--adapter websocket needs --url ws://host/path[/red]")
        raise typer.Exit(2)
    if adapter == "openai_realtime" and not dry_run and not os.environ.get("OPENAI_API_KEY"):
        console.print(
            "[red]--adapter openai_realtime needs OPENAI_API_KEY in the environment[/red]"
        )
        raise typer.Exit(2)
    try:
        backend = get_backend(tts)
    except TTSError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(2) from error

    try:
        scenarios = load_scenarios(target)
    except ScenarioError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    failures = 0
    report_dir = out / datetime.now().strftime("%Y%m%d-%H%M%S")
    settings = {
        "adapter": adapter,
        "tts": tts,
        "seed": seed,
        "realtime": realtime,
        "end_silence_ms": end_silence_ms,
        "mock_endpoint_ms": mock_endpoint_ms,
    }

    for scenario in scenarios:
        cases = expand(
            scenario,
            languages=only_language or None,
            perturbations=only_perturbation or None,
            registers=only_register or None,
            repeats=repeats,
        )
        console.print(f"\n[bold]{scenario.id}[/bold] — {len(cases)} cases — {scenario.description}")
        unused = scenario.unused_variant_languages()
        if unused:
            console.print(f"[yellow]variant languages the matrix never asks for: {unused}[/yellow]")

        if dry_run:
            _print_matrix(cases)
            continue

        logs = asyncio.run(
            _run_cases(
                cases,
                agent=_build_adapter(adapter, url=url, mock_endpoint_ms=mock_endpoint_ms),
                backend=backend,
                seed=seed,
                realtime=realtime,
                end_silence_ms=end_silence_ms,
                log_dir=report_dir / scenario.id / "logs",
            )
        )
        results = [score(log, case) for log, case in zip(logs, cases, strict=True)]
        failures += sum(1 for result in results if not result.passed)

        console.print(
            render_table(
                aggregate(results),
                title="by language and perturbation",
                green=green,
                yellow=yellow,
            )
        )
        console.print(
            render_table(
                aggregate(results, by=("language", "register")),
                title="by register",
                green=green,
                yellow=yellow,
            )
        )
        _print_worst(results)
        paths = write_report(
            results,
            out_dir=report_dir / scenario.id,
            scenario_id=scenario.id,
            settings=settings,
        )
        console.print(f"[dim]report: {paths.markdown}[/dim]")

    if dry_run:
        console.print("\n[dim]--dry-run: no cases were executed[/dim]")
        return
    if failures:
        console.print(f"\n[red]{failures} cases failed[/red]")
        raise typer.Exit(1)
    console.print("\n[green]every case passed[/green]")


@app.command()
def new(
    scenario_id: Annotated[str, typer.Argument(help="Id of the scenario to create.")],
    out: Annotated[Path, typer.Option(help="Directory to write into.")] = Path("."),
) -> None:
    """Write a starter scenario you can fill in."""
    path = out / f"{scenario_id}.yaml"
    if path.exists():
        console.print(f"[red]{path} already exists[/red]")
        raise typer.Exit(1)
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(STARTER.format(id=scenario_id), encoding="utf-8")
    console.print(f"wrote {path}")


@app.command()
def voices(
    backend: Annotated[str, typer.Option(help="Which voice backend to ask.")] = "edge",
    language: Annotated[str | None, typer.Option(help="Only show this locale.")] = None,
) -> None:
    """List the voices a backend can speak with."""
    try:
        found = asyncio.run(get_backend(backend).voices(language))
    except TTSError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(2) from error

    table = Table(title=f"{backend} voices" + (f" for {language}" if language else ""))
    table.add_column("voice")
    table.add_column("language")
    for voice in found:
        table.add_row(voice.id, voice.language)
    console.print(table)


@app.command()
def doctor() -> None:
    """Check that everything a real run needs is present."""
    cache = Path.home() / ".cache" / "sayagain"
    table = Table(title=f"sayagain {__version__}")
    table.add_column("check")
    table.add_column("status")
    table.add_row("python", f"{sys.version_info.major}.{sys.version_info.minor}")
    table.add_row("ffmpeg", _found(shutil.which("ffmpeg")))
    table.add_row("tts cache", f"{cache / 'tts'} ({_cached_utterances(cache)} utterances)")
    table.add_row("tts backends", ", ".join(sorted(BACKENDS)))
    table.add_row("adapters", ", ".join(ADAPTERS))
    table.add_row("edge-tts", _importable("edge_tts"))
    table.add_row("faster-whisper", _importable("faster_whisper"))
    table.add_row("whisper model", _found(_whisper_model_present()))
    table.add_row("openai sdk", _importable("openai"))
    table.add_row("OPENAI_API_KEY", _found(os.environ.get("OPENAI_API_KEY")))
    table.add_row("noise beds", _found(_noise_beds()))
    console.print(table)
    console.print(
        "[dim]a first real run downloads the whisper model and calls the edge-tts "
        "endpoint; both are cached afterwards[/dim]"
    )


def _build_adapter(name: str, *, url: str | None, mock_endpoint_ms: int) -> Adapter:
    """Construct the adapter the user asked for."""
    if name == "websocket":
        return WebSocketAdapter(url or "")
    if name == "openai_realtime":
        return OpenAIRealtimeAdapter()
    return MockAdapter(end_of_turn_silence_ms=mock_endpoint_ms)


async def _run_cases(
    cases: list[Case],
    *,
    agent: Adapter,
    backend: Any,
    seed: int,
    realtime: bool,
    end_silence_ms: int,
    log_dir: Path,
) -> list[RunLog]:
    runner = Runner(
        agent,
        Synthesizer(backend),
        seed=seed,
        realtime=realtime,
        end_silence_ms=end_silence_ms,
        log_dir=log_dir,
    )
    logs: list[RunLog] = []
    for case in track(cases, description="running", console=console):
        logs.append(await runner.run(case))
    return logs


def _print_worst(results: list[CaseResult]) -> None:
    """Show one failing case in full, because a rate alone is not actionable."""
    failures = [result for result in results if not result.passed]
    if not failures:
        return
    worst = failures[0]
    console.print(f"[red]example failure[/red] {worst.case_id}  (locus: {worst.failure_locus})")
    if worst.transcript is not None:
        console.print(f"  heard: {worst.transcript!r}")
    for assertion in worst.assertions:
        if not assertion.passed:
            console.print(f"  [red]x[/red] {assertion.name}: {assertion.detail}")
    if len(failures) > 1:
        console.print(f"  [dim]and {len(failures) - 1} more; see the report[/dim]")


def _print_matrix(cases: list[Case]) -> None:
    table = Table()
    table.add_column("language")
    table.add_column("perturbation")
    table.add_column("cases", justify="right")
    table.add_column("status")
    counts: dict[tuple[str, str], int] = {}
    for case in cases:
        key = (case.language, case.perturbation.id)
        counts[key] = counts.get(key, 0) + 1
    for (language, perturbation), count in counts.items():
        table.add_row(language, perturbation, str(count), "pending")
    console.print(table)


def _cached_utterances(cache: Path) -> int:
    tts = cache / "tts"
    return len(list(tts.glob("*.wav"))) if tts.is_dir() else 0


def _whisper_model_present() -> str | None:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub.is_dir():
        return None
    found = [path for path in hub.glob("models--*whisper*") if path.is_dir()]
    return found[0].name if found else None


def _noise_beds() -> str | None:
    from sayagain.perturb import NOISE_DIR

    beds = sorted(path.stem for path in NOISE_DIR.glob("*.wav"))
    return ", ".join(beds) if beds else None


def _found(value: str | None) -> str:
    return f"[green]{value}[/green]" if value else "[red]missing[/red]"


def _importable(module: str) -> str:
    from importlib.util import find_spec

    return "[green]yes[/green]" if find_spec(module) else "[red]missing[/red]"
