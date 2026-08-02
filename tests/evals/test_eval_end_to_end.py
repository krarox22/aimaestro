"""The harness driving the shipped dataset, with a fake model in place of a real
provider. Proves the pieces fit together without spending a token."""

import pytest
from langchain_core.messages import AIMessage

from aimaestro import graph as graph_module
from aimaestro.schemas import ToDo
from evals.cli import DEFAULT_SCENARIO_DIR, main
from evals.report import aggregate, render
from evals.runner import run_scenario
from evals.scenarios import load_scenarios
from tests.conftest import FakeToolCallingModel, StubExtractor, update_memory_call


def test_shipped_dataset_is_valid():
    """Every scenario file in the repo parses and validates."""
    scenarios = load_scenarios(DEFAULT_SCENARIO_DIR)
    assert len(scenarios) >= 9
    assert {s.layer for s in scenarios} == {"routing", "integrity", "retrieval"}


def test_every_scenario_has_an_expectation():
    """A scenario with no expectations would silently score nothing."""
    for scenario in load_scenarios(DEFAULT_SCENARIO_DIR):
        assert scenario.expect, f"{scenario.id} has no expectations"


def test_every_scenario_declares_matching_expectations():
    """A scenario's layer must match the expectations it actually declares."""
    keys_for = {
        "routing": {"routes"},
        "integrity": {"changed_records", "unchanged", "field_values"},
        "retrieval": {"prompt_contains", "prompt_excludes"},
    }
    for scenario in load_scenarios(DEFAULT_SCENARIO_DIR):
        assert keys_for[scenario.layer] & set(scenario.expect), (
            f"{scenario.id} is layer {scenario.layer} but declares none of "
            f"{keys_for[scenario.layer]}"
        )


def test_retrieval_scenario_scores_green_against_the_real_graph(monkeypatch):
    """A retrieval scenario should pass, because seeded memory really does reach
    the prompt — this exercises seeding, running, and scoring together."""
    monkeypatch.setattr(
        graph_module,
        "get_model",
        lambda model_id: FakeToolCallingModel(responses=[AIMessage(content="hello")]),
    )
    scenario = next(
        s
        for s in load_scenarios(DEFAULT_SCENARIO_DIR)
        if s.id == "retrieval-profile-reaches-the-prompt"
    )
    results = aggregate(scenario, [run_scenario(scenario, model_id="fake:model")])
    assert results[0].layer == "retrieval"
    assert results[0].passes == 1, results[0].failures


def test_integrity_scenario_fails_when_memory_is_mangled(monkeypatch):
    """Point the extractor at a stub that wipes the list and rewrites it, and the
    integrity scorer must catch it. If this test ever passes silently, the
    harness has stopped detecting the failure it exists to detect."""
    # One instance, not one per call: a fresh fake would reset its counter and
    # emit the same tool call forever.
    model = FakeToolCallingModel(
        responses=[update_memory_call("todo"), AIMessage(content="Done.")]
    )
    monkeypatch.setattr(graph_module, "get_model", lambda model_id: model)
    # Returns one record under a brand-new key: the bystanders survive, but the
    # change count is wrong and the target task never gets marked done.
    monkeypatch.setattr(
        graph_module,
        "get_todo_extractor",
        lambda model_id, listener=None: StubExtractor([ToDo(task="something else")]),
    )
    scenario = next(
        s
        for s in load_scenarios(DEFAULT_SCENARIO_DIR)
        if s.id == "integrity-completing-one-leaves-rest-intact"
    )
    results = aggregate(scenario, [run_scenario(scenario, model_id="fake:model")])
    integrity = next(r for r in results if r.layer == "integrity")
    assert integrity.passes == 0
    assert integrity.failures


def test_render_produces_a_readable_report(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "get_model",
        lambda model_id: FakeToolCallingModel(responses=[AIMessage(content="hi")]),
    )
    scenario = next(
        s
        for s in load_scenarios(DEFAULT_SCENARIO_DIR)
        if s.id == "retrieval-profile-reaches-the-prompt"
    )
    text = render(aggregate(scenario, [run_scenario(scenario, model_id="fake:model")]))
    assert "RETRIEVAL" in text
    assert "overall" in text


def test_cli_refuses_to_run_without_an_api_key(monkeypatch, capsys):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("AIMAESTRO_MODEL", "google_genai:gemini-2.5-flash")
    assert main([]) == 1
    assert "GOOGLE_API_KEY" in capsys.readouterr().err


def test_cli_reports_when_no_scenario_matches(monkeypatch, capsys):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyLooksRealEnough")
    assert main(["--scenario", "does-not-exist"]) == 1
    assert "No scenarios matched" in capsys.readouterr().err
