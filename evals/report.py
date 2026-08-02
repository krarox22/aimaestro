"""Aggregate repeated runs into pass rates, and render them.

Rates, not booleans. Tool-calling is not deterministic even at temperature 0, so
a single run tells you almost nothing — a scenario that passes two times in three
is a real and different result from one that always passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evals.runner import RunResult
from evals.scenarios import Scenario
from evals.scoring import SCORERS


@dataclass
class LayerResult:
    scenario_id: str
    layer: str
    passes: int
    runs: int
    failures: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.passes / self.runs if self.runs else 0.0


def aggregate(scenario: Scenario, runs: list[RunResult]) -> list[LayerResult]:
    """Collapse N runs of one scenario into a result per applicable layer."""
    results: list[LayerResult] = []

    for scorer in SCORERS:
        scores = [scorer(scenario, run) for run in runs]
        applicable = [s for s in scores if s.passed is not None]
        if not applicable:
            continue

        failures: list[str] = []
        for score in applicable:
            if not score.passed and score.detail not in failures:
                failures.append(score.detail)

        results.append(
            LayerResult(
                scenario_id=scenario.id,
                layer=applicable[0].layer,
                passes=sum(1 for s in applicable if s.passed),
                runs=len(applicable),
                failures=failures,
            )
        )

    return results


def overall(results: list[LayerResult]) -> tuple[int, int]:
    """Total passes and total scored runs across everything."""
    return sum(r.passes for r in results), sum(r.runs for r in results)


def render(results: list[LayerResult]) -> str:
    """A plain-text report, grouped by layer."""
    if not results:
        return "No scenarios matched.\n"

    lines: list[str] = []
    width = max(len(r.scenario_id) for r in results)

    for layer in ("routing", "integrity", "retrieval"):
        in_layer = [r for r in results if r.layer == layer]
        if not in_layer:
            continue

        lines.append(f"\n{layer.upper()}")
        lines.append("-" * (width + 24))
        for result in sorted(in_layer, key=lambda r: (r.rate, r.scenario_id)):
            pct = round(result.rate * 100)
            mark = "ok  " if result.passes == result.runs else "FAIL"
            lines.append(
                f"  {mark} {result.scenario_id:<{width}}  "
                f"{result.passes}/{result.runs}  {pct:>3}%"
            )
            for failure in result.failures:
                lines.append(f"         └─ {failure}")

    passes, total = overall(results)
    pct = round(passes / total * 100) if total else 0
    lines.append("")
    lines.append("=" * (width + 24))
    lines.append(f"  overall: {passes}/{total} ({pct}%)")
    lines.append("")
    return "\n".join(lines)
