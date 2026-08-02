"""Scoring for the three deterministic layers.

No LLM judge anywhere in here. Everything these functions check is exactly
decidable, which is what makes them trustworthy enough to act on.

A ``Score.passed`` of ``None`` means "not applicable" — the scenario expressed no
expectation for that layer. Not-applicable is never counted as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evals.runner import RunResult
from evals.scenarios import Scenario


@dataclass
class Score:
    layer: str
    passed: bool | None
    detail: str = ""


def _find_by_task(records: dict[str, Any], name: str) -> tuple[str, dict] | None:
    """Locate a todo by task text, tolerating rewording.

    Exact match first; otherwise a case-insensitive substring match either way,
    since a model may legitimately reword "vet appointment" as "Vet appointment
    for dog" while still meaning the same task.
    """
    wanted = name.strip().lower()
    for key, value in records.items():
        if str(value.get("task", "")).strip().lower() == wanted:
            return key, value
    for key, value in records.items():
        actual = str(value.get("task", "")).strip().lower()
        if wanted in actual or actual in wanted:
            return key, value
    return None


# --------------------------------------------------------------------------- #
# Layer 1: routing
# --------------------------------------------------------------------------- #


def score_routing(scenario: Scenario, result: RunResult) -> Score:
    """Did it decide to save, and pick the right memory type?"""
    if "routes" not in scenario.expect:
        return Score("routing", None, "no routing expectation")
    if result.error:
        return Score("routing", False, f"run failed: {result.error}")

    expected = list(scenario.expect["routes"])
    actual = list(result.routes)
    if expected == actual:
        return Score("routing", True, f"routes {actual}")

    if not expected and actual:
        return Score("routing", False, f"saved when it should not have: {actual}")
    if expected and not actual:
        return Score("routing", False, f"did not save; expected {expected}")
    return Score("routing", False, f"expected {expected}, got {actual}")


# --------------------------------------------------------------------------- #
# Layer 2: integrity
# --------------------------------------------------------------------------- #


def score_integrity(scenario: Scenario, result: RunResult) -> Score:
    """Did updating one record leave every other record exactly as it was?"""
    keys = {"changed_records", "unchanged", "field_values"}
    if not keys & set(scenario.expect):
        return Score("integrity", None, "no integrity expectation")
    if result.error:
        return Score("integrity", False, f"run failed: {result.error}")

    seed_todos = result.seed_memory.get("todo", {})
    final_todos = result.final_memory.get("todo", {})
    problems: list[str] = []

    # Records the scenario says must be untouched: compare by their seeded key,
    # so a comparison is against exactly the record it started as.
    for name in scenario.expect.get("unchanged") or []:
        found = _find_by_task(seed_todos, name)
        if found is None:
            problems.append(f"{name!r} was never seeded")
            continue
        key, seeded = found
        if key not in final_todos:
            problems.append(f"{name!r} was deleted")
        elif final_todos[key] != seeded:
            problems.append(
                f"{name!r} was modified: {seeded} -> {final_todos[key]}"
            )

    expected_changes = scenario.expect.get("changed_records")
    if expected_changes is not None:
        changed = sum(
            1
            for key, value in final_todos.items()
            if key not in seed_todos or seed_todos[key] != value
        )
        removed = sum(1 for key in seed_todos if key not in final_todos)
        total = changed + removed
        if total != expected_changes:
            problems.append(
                f"changed {total} record(s), expected {expected_changes}"
            )

    for name, fields in (scenario.expect.get("field_values") or {}).items():
        found = _find_by_task(final_todos, name)
        if found is None:
            problems.append(f"{name!r} missing from final memory")
            continue
        _, value = found
        for field_name, expected_value in fields.items():
            if value.get(field_name) != expected_value:
                problems.append(
                    f"{name!r} {field_name}={value.get(field_name)!r}, "
                    f"expected {expected_value!r}"
                )

    if problems:
        return Score("integrity", False, "; ".join(problems))
    return Score("integrity", True, "records intact")


# --------------------------------------------------------------------------- #
# Layer 3: retrieval
# --------------------------------------------------------------------------- #


def score_retrieval(scenario: Scenario, result: RunResult) -> Score:
    """Did stored memory actually reach the prompt — and only this user's?"""
    keys = {"prompt_contains", "prompt_excludes"}
    if not keys & set(scenario.expect):
        return Score("retrieval", None, "no retrieval expectation")
    if result.error:
        return Score("retrieval", False, f"run failed: {result.error}")

    haystack = "\n".join(result.prompts).lower()
    problems: list[str] = []

    for needle in scenario.expect.get("prompt_contains") or []:
        if str(needle).lower() not in haystack:
            problems.append(f"{needle!r} never reached the prompt")

    for needle in scenario.expect.get("prompt_excludes") or []:
        if str(needle).lower() in haystack:
            problems.append(f"{needle!r} leaked into the prompt")

    if problems:
        return Score("retrieval", False, "; ".join(problems))
    return Score("retrieval", True, "memory reached the prompt")


SCORERS = (score_routing, score_integrity, score_retrieval)


def score_all(scenario: Scenario, result: RunResult) -> list[Score]:
    """Every applicable score for one run."""
    scores = [scorer(scenario, result) for scorer in SCORERS]
    return [s for s in scores if s.passed is not None]
