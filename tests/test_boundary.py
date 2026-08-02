"""The boundary between the test suite and the eval harness.

Tests assert deterministic behavior against fakes and are free to run. Evals
score a real model and cost money. Both are importable from pytest, so this
file enforces the separation rather than trusting it.
"""

import pytest

from aimaestro.graph import get_model
from evals.cli import DEFAULT_SCENARIO_DIR
from evals.runner import run_scenario
from evals.scenarios import load_scenarios
from tests.conftest import RealModelForbidden


def test_building_a_real_model_is_blocked_in_tests():
    """The guard fires, so a careless test cannot silently start billing."""
    get_model.cache_clear()
    with pytest.raises(RealModelForbidden, match="real model"):
        get_model("google_genai:gemini-2.5-flash")


def test_guard_names_the_right_escape_hatch():
    get_model.cache_clear()
    with pytest.raises(RealModelForbidden, match=r"python -m evals"):
        get_model("openai:gpt-4o")


def test_guard_survives_the_runners_catch_all(monkeypatch):
    """The runner swallows every Exception so a broken run becomes data. This
    guard must sail through that, or a test that spends money reports green.

    Regression: it originally derived from Exception and was silently recorded
    as run data instead of failing the test.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyLooksCompletelyReal123")
    scenario = next(
        s
        for s in load_scenarios(DEFAULT_SCENARIO_DIR)
        if s.id == "routing-saves-a-new-task"
    )
    with pytest.raises(RealModelForbidden):
        run_scenario(scenario, model_id="google_genai:gemini-2.5-flash")


def test_eval_harness_does_not_depend_on_the_test_suite():
    """`evals/` must stand alone. Importing test helpers would tangle the two
    tools together and make the harness undeployable on its own."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "evals"
    offenders = [
        path.name
        for path in root.rglob("*.py")
        if "import tests" in path.read_text() or "from tests" in path.read_text()
    ]
    assert offenders == []
