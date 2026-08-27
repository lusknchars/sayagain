"""Expand a scenario into the case matrix.

One intent becomes many cases: register x language x perturbation x repeat.
A register that has no line in a given language is skipped rather than failed —
partial translations are the normal state of a scenario that is being grown.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sayagain.scenario import PerturbationSpec, Scenario


@dataclass(frozen=True, slots=True)
class Case:
    """One run: one surface form of one intent, under one perturbation."""

    scenario: Scenario
    register: str
    language: str
    perturbation: PerturbationSpec
    repeat: int
    turn_texts: list[str] = field(default_factory=list)
    interrupt_texts: list[str | None] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Stable identifier, used for report rows and cache keys."""
        return "/".join(
            (
                self.scenario.id,
                self.language,
                self.perturbation.id,
                self.register,
                str(self.repeat),
            )
        )


def expand(
    scenario: Scenario,
    *,
    languages: Sequence[str] | None = None,
    perturbations: Sequence[str] | None = None,
    registers: Sequence[str] | None = None,
    repeats: int | None = None,
) -> list[Case]:
    """Build every case a scenario asks for, after applying the CLI filters."""
    wanted_languages = [
        language
        for language in scenario.matrix.language
        if languages is None or language in languages
    ]
    wanted_perturbations = [
        perturbation
        for perturbation in scenario.matrix.perturbation
        if perturbations is None or perturbation.id in perturbations
    ]
    all_registers = sorted({register for turn in scenario.turns for register in turn.user.variants})
    wanted_registers = [
        register for register in all_registers if registers is None or register in registers
    ]
    total_repeats = repeats if repeats is not None else scenario.repeats

    cases: list[Case] = []
    for language in wanted_languages:
        for register in wanted_registers:
            texts = _resolve_turns(scenario, register, language)
            if texts is None:
                continue
            turn_texts, interrupt_texts = texts
            for perturbation in wanted_perturbations:
                for repeat in range(1, total_repeats + 1):
                    cases.append(
                        Case(
                            scenario=scenario,
                            register=register,
                            language=language,
                            perturbation=perturbation,
                            repeat=repeat,
                            turn_texts=turn_texts,
                            interrupt_texts=interrupt_texts,
                        )
                    )
    return cases


def _resolve_turns(
    scenario: Scenario, register: str, language: str
) -> tuple[list[str], list[str | None]] | None:
    """Return the lines for one register+language, or None if any turn is missing."""
    turn_texts: list[str] = []
    interrupt_texts: list[str | None] = []
    for turn in scenario.turns:
        by_language = turn.user.variants.get(register)
        if by_language is None or language not in by_language:
            return None
        turn_texts.append(by_language[language])
        # A barge-in with no line in this language is dropped; the case still runs.
        interrupt = turn.interrupt.text.get(language) if turn.interrupt else None
        interrupt_texts.append(interrupt)
    return turn_texts, interrupt_texts
