"""Scenario files: the unit of evaluation for a stateful memory agent.

A scenario is not an input/output pair. It is a starting memory state, a
sequence of turns, and expectations about the memory state that results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LAYERS = ("routing", "integrity", "retrieval")

#: Namespaces a scenario may seed.
SEED_KEYS = ("profile", "todo", "instructions")

#: Expectation keys, and which scorer consumes each.
EXPECT_KEYS = (
    "routes",  # routing
    "changed_records",  # integrity
    "unchanged",  # integrity
    "field_values",  # integrity
    "prompt_contains",  # retrieval
    "prompt_excludes",  # retrieval
)


class ScenarioError(Exception):
    """A scenario file is malformed."""


@dataclass
class Scenario:
    id: str
    layer: str
    turns: list[str]
    description: str = ""
    seed: dict[str, Any] = field(default_factory=dict)
    expect: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None


def _validate(raw: dict, source: str) -> None:
    if not isinstance(raw, dict):
        raise ScenarioError(f"{source}: expected a mapping at the top level")

    for required in ("id", "layer", "turns"):
        if not raw.get(required):
            raise ScenarioError(f"{source}: missing required key {required!r}")

    if raw["layer"] not in LAYERS:
        raise ScenarioError(
            f"{source}: unknown layer {raw['layer']!r}; expected one of {LAYERS}"
        )

    if not isinstance(raw["turns"], list):
        raise ScenarioError(f"{source}: 'turns' must be a list of messages")

    # Typos in expectations would otherwise score as a silent pass.
    for key in raw.get("expect") or {}:
        if key not in EXPECT_KEYS:
            raise ScenarioError(
                f"{source}: unknown expect key {key!r}; expected one of {EXPECT_KEYS}"
            )

    for key in raw.get("seed") or {}:
        if key not in SEED_KEYS:
            raise ScenarioError(
                f"{source}: unknown seed key {key!r}; expected one of {SEED_KEYS}"
            )


def load_scenario(path: str | Path) -> Scenario:
    """Parse and validate one scenario file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    _validate(raw, path.name)
    return Scenario(
        id=raw["id"],
        layer=raw["layer"],
        turns=list(raw["turns"]),
        description=raw.get("description", ""),
        seed=raw.get("seed") or {},
        expect=raw.get("expect") or {},
        path=path,
    )


def load_scenarios(directory: str | Path) -> list[Scenario]:
    """Load every scenario in a directory tree, sorted by id."""
    directory = Path(directory)
    scenarios: list[Scenario] = []
    seen: dict[str, Path] = {}

    for path in sorted(directory.rglob("*.yaml")):
        scenario = load_scenario(path)
        if scenario.id in seen:
            raise ScenarioError(
                f"duplicate scenario id {scenario.id!r} in {path.name} "
                f"and {seen[scenario.id].name}"
            )
        seen[scenario.id] = path
        scenarios.append(scenario)

    return sorted(scenarios, key=lambda s: s.id)
