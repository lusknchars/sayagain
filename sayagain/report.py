"""Report artifacts: terminal table, report.json, report.md, heatmap.png.

The ordering here is a deliberate product decision. A pass rate tells you
nothing you can act on, and a single failing assertion can zero out the whole
board while three others are fine. So the markdown leads with the utterances
that broke and what the agent did with them, and every rate is reported per
assertion rather than collapsed into one number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.table import Table

from sayagain.score import Aggregate, CaseResult, aggregate

#: Assertion order in tables, most diagnostic first.
ASSERTION_ORDER = ("tool_call", "end_state", "max_first_audio_ms", "max_barge_in_stop_ms", "ran")


@dataclass(frozen=True, slots=True)
class ReportPaths:
    """Where the artifacts landed."""

    directory: Path
    json: Path
    markdown: Path
    heatmap: Path | None


def write_report(
    results: list[CaseResult],
    *,
    out_dir: Path,
    scenario_id: str,
    settings: dict[str, Any] | None = None,
) -> ReportPaths:
    """Write report.json, report.md and heatmap.png into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_condition = aggregate(results, by=("language", "perturbation"))
    by_register = aggregate(results, by=("language", "register"))

    payload = {
        "scenario_id": scenario_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "settings": settings or {},
        "totals": {
            "cases": len(results),
            "passed": sum(1 for result in results if result.passed),
            "failed": sum(1 for result in results if not result.passed),
        },
        "by_language_and_perturbation": [_row_json(row) for row in by_condition],
        "by_language_and_register": [_row_json(row) for row in by_register],
        "cases": [_case_json(result) for result in results],
    }
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_path = out_dir / "report.md"
    markdown_path.write_text(
        _markdown(results, by_condition, by_register, scenario_id, settings or {}),
        encoding="utf-8",
    )

    heatmap_path = out_dir / "heatmap.png"
    drawn = _heatmap(results, heatmap_path, scenario_id)

    return ReportPaths(
        directory=out_dir,
        json=json_path,
        markdown=markdown_path,
        heatmap=heatmap_path if drawn else None,
    )


def render_table(rows: list[Aggregate], *, title: str, green: float, yellow: float) -> Table:
    """Build the terminal table: one column per assertion, not one pass rate."""
    names = _assertion_names(rows)
    table = Table(title=title)
    for axis in rows[0].labels if rows else {"language": ""}:
        table.add_column(axis)
    table.add_column("cases", justify="right")
    for name in names:
        table.add_column(_short(name), justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("top failure")

    for row in sorted(rows, key=lambda item: tuple(item.labels.values())):
        cells = [*row.labels.values(), str(row.cases)]
        for name in names:
            rate = row.assertion_rates.get(name)
            cells.append("-" if rate is None else _coloured(rate, green, yellow))
        cells.append(_ms(row.p50_first_audio_ms))
        cells.append(row.top_failure_locus)
        table.add_row(*cells)
    return table


def _markdown(
    results: list[CaseResult],
    by_condition: list[Aggregate],
    by_register: list[Aggregate],
    scenario_id: str,
    settings: dict[str, Any],
) -> str:
    failures = [result for result in results if not result.passed]
    lines = [
        f"# {scenario_id}",
        "",
        f"{len(results)} cases, {len(results) - len(failures)} passed, {len(failures)} failed.",
        "",
    ]
    if settings:
        lines += ["Run with " + ", ".join(f"`{k}={v}`" for k, v in settings.items()) + ".", ""]

    if failures:
        lines += ["## What broke", ""]
        lines += _worst_cases(failures)
    else:
        lines += ["Every case passed.", ""]

    lines += ["## Pass rate by assertion", ""]
    lines += _markdown_table(by_condition)
    lines += ["", "## Pass rate by register", ""]
    lines += _markdown_table(by_register)
    lines += [
        "",
        "## Reading this",
        "",
        "Rates are per assertion. One failing assertion does not zero the others,",
        "because knowing *which* of four things broke is the whole point.",
        "`failure_locus` says where: `asr` means the words never arrived,",
        "`reasoning` means they arrived and were misused, `unknown` means the",
        "agent reported no transcript so the question cannot be answered.",
        "",
    ]
    return "\n".join(lines)


def _worst_cases(failures: list[CaseResult]) -> list[str]:
    """One failing case per condition, in full. This is the actionable part."""
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for result in failures:
        key = (result.language, result.perturbation)
        if key in seen:
            continue
        seen.add(key)
        lines += [
            f"### {result.language} / {result.perturbation} / {result.register}",
            "",
            f"- locus: **{result.failure_locus}**",
        ]
        if result.transcript is not None:
            lines.append(f"- the agent heard: `{result.transcript}`")
        if result.transcript_wer is not None:
            lines.append(f"- word error rate: {result.transcript_wer:.0%}")
        for assertion in result.assertions:
            if not assertion.passed:
                lines.append(f"- **{assertion.name}**: {assertion.detail}")
        lines += [f"- replay: `{result.case_id}`", ""]
    remaining = len(failures) - len(seen)
    if remaining > 0:
        lines += [f"{remaining} further failures are in `report.json` and the JSONL logs.", ""]
    return lines


def _markdown_table(rows: list[Aggregate]) -> list[str]:
    if not rows:
        return ["_nothing ran._"]
    names = _assertion_names(rows)
    axes = list(rows[0].labels)
    header = [*axes, "cases", *[_short(name) for name in names], "p50 ms", "top failure"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in sorted(rows, key=lambda item: tuple(item.labels.values())):
        cells = [*row.labels.values(), str(row.cases)]
        for name in names:
            rate = row.assertion_rates.get(name)
            cells.append("-" if rate is None else f"{rate:.0%}")
        cells += [_ms(row.p50_first_audio_ms), row.top_failure_locus]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _heatmap(results: list[CaseResult], path: Path, scenario_id: str) -> bool:
    """Draw one panel per assertion: rows are language/register, columns perturbation.

    Rows are registers rather than languages alone because that is the axis the
    data actually moves along; a single all-red square teaches nothing.
    """
    if not results:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [
        name
        for name in ASSERTION_ORDER
        if any(assertion.name == name for result in results for assertion in result.assertions)
    ]
    rows = sorted({f"{result.language} {result.register}" for result in results})
    columns = sorted({result.perturbation for result in results})
    if not names or not rows or not columns:
        return False

    figure, axes = plt.subplots(
        1, len(names), figsize=(3.1 * len(names) + 1.6, 0.46 * len(rows) + 2.1), squeeze=False
    )
    for index, name in enumerate(names):
        grid = np.full((len(rows), len(columns)), np.nan)
        for row_index, row in enumerate(rows):
            for column_index, column in enumerate(columns):
                subset = [
                    assertion.passed
                    for result in results
                    if f"{result.language} {result.register}" == row
                    and result.perturbation == column
                    for assertion in result.assertions
                    if assertion.name == name
                ]
                if subset:
                    grid[row_index, column_index] = sum(subset) / len(subset)
        axis = axes[0][index]
        axis.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_title(_short(name), fontsize=10)
        axis.set_xticks(range(len(columns)), columns, rotation=45, ha="right", fontsize=8)
        if index == 0:
            axis.set_yticks(range(len(rows)), rows, fontsize=8)
        else:
            axis.set_yticks([])
        for row_index in range(len(rows)):
            for column_index in range(len(columns)):
                value = grid[row_index, column_index]
                if not np.isnan(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.0%}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )
    figure.suptitle(f"{scenario_id} — pass rate per assertion", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def _row_json(row: Aggregate) -> dict[str, Any]:
    return {
        "labels": row.labels,
        "cases": row.cases,
        "pass_rate": row.pass_rate,
        "pass_at_k": row.pass_at_k,
        "pass_hat_k": row.pass_hat_k,
        "assertion_rates": row.assertion_rates,
        "locus_counts": row.locus_counts,
        "p50_first_audio_ms": row.p50_first_audio_ms,
        "p95_first_audio_ms": row.p95_first_audio_ms,
        "p50_barge_in_stop_ms": row.p50_barge_in_stop_ms,
        "p95_barge_in_stop_ms": row.p95_barge_in_stop_ms,
    }


def _case_json(result: CaseResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "language": result.language,
        "register": result.register,
        "perturbation": result.perturbation,
        "repeat": result.repeat,
        "passed": result.passed,
        "failure_locus": result.failure_locus,
        "first_audio_ms": result.first_audio_ms,
        "barge_in_stop_ms": result.barge_in_stop_ms,
        "tool_call_correct": result.tool_call_correct,
        "end_state_correct": result.end_state_correct,
        "transcript_wer": result.transcript_wer,
        "transcript": result.transcript,
        "error": result.error,
        "assertions": [
            {"name": a.name, "passed": a.passed, "detail": a.detail} for a in result.assertions
        ],
    }


def _assertion_names(rows: list[Aggregate]) -> list[str]:
    seen = {name for row in rows for name in row.assertion_rates}
    ordered = [name for name in ASSERTION_ORDER if name in seen]
    return ordered + sorted(seen - set(ordered))


def _short(name: str) -> str:
    return {
        "tool_call": "tool",
        "end_state": "state",
        "max_first_audio_ms": "latency",
        "max_barge_in_stop_ms": "barge-in",
        "ran": "ran",
    }.get(name, name)


def _coloured(rate: float, green: float, yellow: float) -> str:
    colour = "green" if rate >= green else "yellow" if rate >= yellow else "red"
    return f"[{colour}]{rate:.0%}[/{colour}]"


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"
