import pytest
from langchain_core.messages import AIMessage

from aimaestro import graph as graph_module
from aimaestro.schemas import ToDo
from evals.runner import RunResult, run_scenario
from evals.scenarios import Scenario
from tests.conftest import FakeToolCallingModel, StubExtractor, update_memory_call


@pytest.fixture
def fake_model(monkeypatch):
    def install(responses):
        model = FakeToolCallingModel(responses=responses)
        monkeypatch.setattr(graph_module, "get_model", lambda model_id: model)
        return model

    return install


def _scenario(**kwargs):
    base = dict(id="s", layer="routing", turns=["hello"], seed={}, expect={})
    base.update(kwargs)
    return Scenario(**base)


def test_seeds_profile_todo_and_instructions(fake_model):
    fake_model([AIMessage(content="hi")])
    scenario = _scenario(
        seed={
            "profile": {"name": "Kratika"},
            "todo": [{"task": "renew passport", "status": "not started"}],
            "instructions": "always add deadlines",
        }
    )
    result = run_scenario(scenario, model_id="fake:model")

    assert result.seed_memory["profile"]["seed-profile"]["name"] == "Kratika"
    assert result.seed_memory["todo"]["seed-todo-0"]["task"] == "renew passport"
    assert "deadlines" in result.seed_memory["instructions"]["user_instructions"]["memory"]


def test_seeded_memory_reaches_the_prompt(fake_model):
    fake_model([AIMessage(content="hi")])
    scenario = _scenario(seed={"profile": {"name": "Kratika"}})
    result = run_scenario(scenario, model_id="fake:model")

    assert result.prompts
    assert "Kratika" in result.prompts[0]


def test_captures_routes_taken(fake_model, monkeypatch):
    fake_model([update_memory_call("todo"), AIMessage(content="Added.")])
    monkeypatch.setattr(
        graph_module,
        "get_todo_extractor",
        lambda model_id, listener=None: StubExtractor([ToDo(task="book vet")]),
    )
    result = run_scenario(_scenario(turns=["book the vet"]), model_id="fake:model")
    assert result.routes == ["update_todos"]


def test_no_tool_call_means_no_routes(fake_model):
    fake_model([AIMessage(content="just chatting")])
    result = run_scenario(_scenario(), model_id="fake:model")
    assert result.routes == []


def test_captures_reply_per_turn(fake_model):
    fake_model([AIMessage(content="first"), AIMessage(content="second")])
    result = run_scenario(_scenario(turns=["a", "b"]), model_id="fake:model")
    assert result.replies == ["first", "second"]


def test_final_memory_reflects_writes(fake_model, monkeypatch):
    fake_model([update_memory_call("todo"), AIMessage(content="Added.")])
    monkeypatch.setattr(
        graph_module,
        "get_todo_extractor",
        lambda model_id, listener=None: StubExtractor([ToDo(task="book vet")]),
    )
    result = run_scenario(_scenario(turns=["book the vet"]), model_id="fake:model")
    tasks = [v["task"] for v in result.final_memory["todo"].values()]
    assert "book vet" in tasks


def test_runs_are_isolated_from_each_other(fake_model, monkeypatch):
    """Each run gets a fresh database.

    If runs shared one, the second would find two todos instead of one — which
    would quietly corrupt every pass rate the harness reports.
    """
    monkeypatch.setattr(
        graph_module,
        "get_todo_extractor",
        lambda model_id, listener=None: StubExtractor([ToDo(task="book vet")]),
    )
    scenario = _scenario(turns=["book the vet"])
    script = [update_memory_call("todo"), AIMessage(content="Added.")]

    fake_model(list(script))
    first = run_scenario(scenario, model_id="fake:model")

    fake_model(list(script))  # a real model would not need resetting
    second = run_scenario(scenario, model_id="fake:model")

    assert len(first.final_memory["todo"]) == 1
    assert len(second.final_memory["todo"]) == 1


@pytest.mark.timeout(30)
def test_runaway_save_loop_terminates(monkeypatch):
    """A model that never stops calling UpdateMemory must fail, not hang.

    The recursion cap is what makes an eval run bounded; without it one
    pathological scenario stalls the whole dataset indefinitely.
    """

    def always_saving(model_id):
        # A fresh instance per call, so its script never advances.
        return FakeToolCallingModel(responses=[update_memory_call("todo")])

    monkeypatch.setattr(graph_module, "get_model", always_saving)
    monkeypatch.setattr(
        graph_module,
        "get_todo_extractor",
        lambda model_id, listener=None: StubExtractor([ToDo(task="looping")]),
    )
    result = run_scenario(_scenario(turns=["go"]), model_id="fake:model")
    assert result.error is not None
    assert "recursion" in result.error.lower() or "limit" in result.error.lower()


def test_error_is_recorded_not_raised(monkeypatch):
    def explode(model_id):
        raise RuntimeError("model unreachable")

    monkeypatch.setattr(graph_module, "get_model", explode)
    result = run_scenario(_scenario(), model_id="fake:model")
    assert isinstance(result, RunResult)
    assert result.error is not None
    assert "unreachable" in result.error
