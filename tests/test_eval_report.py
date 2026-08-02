from pytest import approx

from evals.report import LayerResult, aggregate, overall, render
from evals.runner import RunResult
from evals.scenarios import Scenario
from evals.scoring import Score


def _scenario(expect):
    return Scenario(id="s", layer="routing", turns=["x"], expect=expect)


def test_aggregate_computes_pass_rate():
    scenario = _scenario({"routes": ["update_todos"]})
    runs = [
        RunResult(scenario_id="s", routes=["update_todos"]),
        RunResult(scenario_id="s", routes=["update_todos"]),
        RunResult(scenario_id="s", routes=["update_profile"]),
    ]
    results = aggregate(scenario, runs)
    assert len(results) == 1
    assert results[0].passes == 2
    assert results[0].runs == 3
    assert results[0].rate == approx(2 / 3)


def test_aggregate_records_distinct_failure_details():
    scenario = _scenario({"routes": ["update_todos"]})
    runs = [
        RunResult(scenario_id="s", routes=["update_profile"]),
        RunResult(scenario_id="s", routes=["update_profile"]),
    ]
    results = aggregate(scenario, runs)
    assert len(results[0].failures) == 1  # deduplicated


def test_aggregate_skips_layers_with_no_expectation():
    scenario = _scenario({"routes": ["update_todos"]})
    runs = [RunResult(scenario_id="s", routes=["update_todos"])]
    assert [r.layer for r in aggregate(scenario, runs)] == ["routing"]


def test_overall_counts_across_scenarios():
    results = [
        LayerResult("a", "routing", passes=3, runs=3, failures=[]),
        LayerResult("b", "integrity", passes=1, runs=3, failures=["x"]),
    ]
    passes, total = overall(results)
    assert (passes, total) == (4, 6)


def test_render_includes_ids_rates_and_failures():
    results = [
        LayerResult("scenario-a", "routing", passes=3, runs=3, failures=[]),
        LayerResult("scenario-b", "integrity", passes=1, runs=3, failures=["mangled"]),
    ]
    text = render(results)
    assert "scenario-a" in text
    assert "scenario-b" in text
    assert "100" in text
    assert "33" in text
    assert "mangled" in text


def test_render_handles_no_results():
    assert render([]) .strip() != ""


def test_full_pass_and_partial_pass_are_distinguishable():
    """A pass rate below 1.0 must be visible, not rounded away."""
    results = [LayerResult("s", "routing", passes=2, runs=3, failures=["once"])]
    text = render(results)
    assert "67" in text or "66" in text
