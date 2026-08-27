"""Scenario schema: the YAML a user writes, parsed and validated.

A scenario holds one intent constant and declares the surface forms it should
survive: languages, registers, and acoustic perturbations. Everything that can
be a typo in YAML is validated here, because a typo that slips through becomes
a case that silently never runs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

#: BCP-47-ish tags, deliberately narrow: `pt-BR`, not `pt_BR` or `pt-br`.
LANGUAGE_TAG = re.compile(r"^[a-z]{2,3}(-[A-Z]{2})?$")


class ScenarioError(Exception):
    """A scenario file could not be read, parsed, or validated."""


class _Base(BaseModel):
    """Shared config: unknown keys are errors, aliases may be used by field name."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ToolSpec(_Base):
    """A tool the agent under test is expected to expose."""

    name: str
    # `schema` in YAML; renamed here because it shadows a pydantic BaseModel attribute.
    parameters: dict[str, str] = Field(default_factory=dict, alias="schema")


class AgentSpec(_Base):
    """How to configure the agent under test for this scenario."""

    system_prompt: str | None = None
    system_prompt_file: Path | None = None
    tools: list[ToolSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _at_most_one_prompt_form(self) -> AgentSpec:
        if self.system_prompt is not None and self.system_prompt_file is not None:
            raise ValueError("set either system_prompt or system_prompt_file, not both")
        return self

    @property
    def tool_names(self) -> set[str]:
        """Names of every declared tool."""
        return {tool.name for tool in self.tools}


class PerturbationSpec(_Base):
    """One acoustic perturbation: a preset id plus optional overrides."""

    id: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"id": value}
        return value


class Matrix(_Base):
    """The axes a single intent is expanded across."""

    language: list[str] = Field(min_length=1)
    voice: Literal["default"] | dict[str, list[str]] = "default"
    perturbation: list[PerturbationSpec] = Field(
        default_factory=lambda: [PerturbationSpec(id="clean")],
        min_length=1,
    )

    @field_validator("language")
    @classmethod
    def _check_language_tags(cls, value: list[str]) -> list[str]:
        for tag in value:
            if not LANGUAGE_TAG.match(tag):
                raise ValueError(f"{tag!r} is not a language tag like 'pt-BR'")
        duplicates = sorted({tag for tag in value if value.count(tag) > 1})
        if duplicates:
            raise ValueError(f"duplicate languages would double the matrix: {duplicates}")
        return value

    @field_validator("voice", mode="before")
    @classmethod
    def _normalise_voice(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {lang: [v] if isinstance(v, str) else v for lang, v in value.items()}
        return value

    def voices_for(self, language: str) -> list[str] | None:
        """Return the voice ids to use for `language`, or None to let the TTS layer pick."""
        if isinstance(self.voice, str):
            return None
        return self.voice.get(language)


class UserTurn(_Base):
    """One user intent and the surface forms that express it."""

    intent: str
    #: register -> language -> utterance
    variants: dict[str, dict[str, str]] = Field(min_length=1)

    @field_validator("variants")
    @classmethod
    def _check_variant_languages(
        cls, value: dict[str, dict[str, str]]
    ) -> dict[str, dict[str, str]]:
        for register, by_language in value.items():
            if not by_language:
                raise ValueError(f"register {register!r} has no utterances")
            for tag in by_language:
                if not LANGUAGE_TAG.match(tag):
                    raise ValueError(f"{tag!r} is not a language tag like 'pt-BR'")
        return value


class InterruptSpec(_Base):
    """A barge-in: speak over the agent after it has been talking for a while."""

    after_agent_speaks_ms: int = Field(gt=0)
    # `with` in YAML; renamed here because `with` is a Python keyword.
    text: dict[str, str] = Field(alias="with", min_length=1)


class ExpectedToolCall(_Base):
    """The tool call the agent should make, matched through the normalizer."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Expect(_Base):
    """What has to be true for a case to pass."""

    tool_call: ExpectedToolCall | None = None
    max_first_audio_ms: int | None = Field(default=None, gt=0)
    max_barge_in_stop_ms: int | None = Field(default=None, gt=0)
    end_state: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _at_least_one_expectation(self) -> Expect:
        if not any(
            (
                self.tool_call,
                self.max_first_audio_ms,
                self.max_barge_in_stop_ms,
                self.end_state,
            )
        ):
            raise ValueError("expect declares nothing, so the case can never fail")
        return self


class Turn(_Base):
    """One exchange: what the user says, optional barge-in, what must happen."""

    user: UserTurn
    interrupt: InterruptSpec | None = None
    expect: Expect


class Scenario(_Base):
    """A whole scenario file."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str
    agent: AgentSpec
    matrix: Matrix
    repeats: int = Field(default=1, ge=1)
    turns: list[Turn] = Field(min_length=1)
    #: Set by the loader; relative paths in the file resolve against its parent.
    source_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _expected_tools_are_declared(self) -> Scenario:
        declared = self.agent.tool_names
        for index, turn in enumerate(self.turns, start=1):
            expected = turn.expect.tool_call
            if expected is not None and expected.name not in declared:
                raise ValueError(
                    f"turn {index} expects tool {expected.name!r}, "
                    f"which agent.tools does not declare ({sorted(declared)})"
                )
        return self

    @model_validator(mode="after")
    def _turns_cover_the_matrix(self) -> Scenario:
        languages = set(self.matrix.language)
        for index, turn in enumerate(self.turns, start=1):
            covered = {tag for by_language in turn.user.variants.values() for tag in by_language}
            if not covered & languages:
                raise ValueError(
                    f"turn {index}: no variant covers any language in the matrix "
                    f"({sorted(languages)}); it would expand to zero cases"
                )
        return self

    def system_prompt_path(self) -> Path | None:
        """Absolute path to the prompt file, or None if the prompt is inline or absent."""
        if self.agent.system_prompt_file is None:
            return None
        base = self.source_path.parent if self.source_path else Path.cwd()
        return (base / self.agent.system_prompt_file).resolve()

    def system_prompt(self) -> str | None:
        """Return the system prompt text, reading the file when the scenario points at one."""
        if self.agent.system_prompt is not None:
            return self.agent.system_prompt
        path = self.system_prompt_path()
        if path is None:
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ScenarioError(f"{self.id}: cannot read system_prompt_file: {exc}") from exc

    def unused_variant_languages(self) -> list[str]:
        """Languages written in the file that the matrix never asks for.

        Not an error — extra translations are harmless — but almost always a
        typo, so `sayagain doctor` reports them.
        """
        declared = set(self.matrix.language)
        seen: set[str] = set()
        for turn in self.turns:
            for by_language in turn.user.variants.values():
                seen |= set(by_language)
            if turn.interrupt is not None:
                seen |= set(turn.interrupt.text)
        return sorted(seen - declared)


def load_scenario(path: Path | str) -> Scenario:
    """Load and validate one scenario file.

    Raises:
        ScenarioError: the file is unreadable, is not valid YAML, or fails validation.

    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ScenarioError(f"{path}: cannot read scenario: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path}: invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ScenarioError(f"{path}: expected a YAML mapping at the top level")

    try:
        scenario = Scenario.model_validate({**raw, "source_path": path})
    except ValidationError as exc:
        raise ScenarioError(_format_validation_error(path, exc)) from exc

    prompt_path = scenario.system_prompt_path()
    if prompt_path is not None and not prompt_path.is_file():
        raise ScenarioError(
            f"{path}: system_prompt_file not found: {scenario.agent.system_prompt_file}"
        )
    return scenario


def load_scenarios(target: Path | str) -> list[Scenario]:
    """Load one scenario file, or every `*.yaml` / `*.yml` directly inside a directory.

    Raises:
        ScenarioError: the target does not exist, or a directory holds no scenarios.

    """
    target = Path(target)
    if target.is_file():
        return [load_scenario(target)]
    if not target.is_dir():
        raise ScenarioError(f"{target}: no such file or directory")

    paths = sorted(p for p in target.iterdir() if p.suffix in {".yaml", ".yml"} and p.is_file())
    if not paths:
        raise ScenarioError(f"{target}: no scenarios found (looked for *.yaml and *.yml)")
    return [load_scenario(p) for p in paths]


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"{path}: invalid scenario"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)
