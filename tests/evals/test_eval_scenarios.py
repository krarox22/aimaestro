import pytest

from evals.scenarios import Scenario, ScenarioError, load_scenario, load_scenarios

VALID = """
id: marking-one-done-leaves-rest-intact
layer: integrity
description: Marking one task done must not disturb the others.
seed:
  todo:
    - {task: vet appointment, status: not started}
    - {task: renew passport, status: not started}
turns:
  - the vet appointment is done
expect:
  changed_records: 1
  unchanged: [renew passport]
  field_values:
    vet appointment: {status: done}
"""


def _write(tmp_path, text, name="s.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_valid_scenario_parses(tmp_path):
    s = load_scenario(_write(tmp_path, VALID))
    assert isinstance(s, Scenario)
    assert s.id == "marking-one-done-leaves-rest-intact"
    assert s.layer == "integrity"
    assert s.turns == ["the vet appointment is done"]
    assert s.seed["todo"][0]["task"] == "vet appointment"
    assert s.expect["changed_records"] == 1


def test_missing_id_raises(tmp_path):
    with pytest.raises(ScenarioError, match="id"):
        load_scenario(_write(tmp_path, "layer: routing\nturns: [hi]\n"))


def test_missing_turns_raises(tmp_path):
    with pytest.raises(ScenarioError, match="turns"):
        load_scenario(_write(tmp_path, "id: x\nlayer: routing\n"))


def test_unknown_layer_raises(tmp_path):
    with pytest.raises(ScenarioError, match="layer"):
        load_scenario(_write(tmp_path, "id: x\nlayer: telepathy\nturns: [hi]\n"))


def test_unknown_expect_key_raises(tmp_path):
    """A typo in an expectation must fail loudly, not silently score as a pass."""
    text = "id: x\nlayer: routing\nturns: [hi]\nexpect: {rout: [update_todos]}\n"
    with pytest.raises(ScenarioError, match="rout"):
        load_scenario(_write(tmp_path, text))


def test_unknown_seed_namespace_raises(tmp_path):
    text = "id: x\nlayer: routing\nturns: [hi]\nseed: {profil: {}}\n"
    with pytest.raises(ScenarioError, match="profil"):
        load_scenario(_write(tmp_path, text))


def test_seed_and_expect_default_to_empty(tmp_path):
    s = load_scenario(_write(tmp_path, "id: x\nlayer: routing\nturns: [hi]\n"))
    assert s.seed == {}
    assert s.expect == {}


def test_load_directory_finds_all(tmp_path):
    _write(tmp_path, VALID, "a.yaml")
    _write(tmp_path, "id: second\nlayer: routing\nturns: [hi]\n", "b.yaml")
    scenarios = load_scenarios(tmp_path)
    assert {s.id for s in scenarios} == {
        "marking-one-done-leaves-rest-intact",
        "second",
    }


def test_duplicate_ids_raise(tmp_path):
    _write(tmp_path, "id: dupe\nlayer: routing\nturns: [hi]\n", "a.yaml")
    _write(tmp_path, "id: dupe\nlayer: routing\nturns: [hi]\n", "b.yaml")
    with pytest.raises(ScenarioError, match="dupe"):
        load_scenarios(tmp_path)


def test_scenarios_are_sorted_by_id(tmp_path):
    _write(tmp_path, "id: zebra\nlayer: routing\nturns: [hi]\n", "z.yaml")
    _write(tmp_path, "id: apple\nlayer: routing\nturns: [hi]\n", "a.yaml")
    assert [s.id for s in load_scenarios(tmp_path)] == ["apple", "zebra"]
