from evals.runner import RunResult
from evals.scenarios import Scenario
from evals.scoring import score_all, score_integrity, score_retrieval, score_routing


def _scenario(expect, layer="routing", seed=None):
    return Scenario(
        id="s", layer=layer, turns=["x"], seed=seed or {}, expect=expect
    )


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def test_routing_passes_on_match():
    scenario = _scenario({"routes": ["update_todos"]})
    result = RunResult(scenario_id="s", routes=["update_todos"])
    assert score_routing(scenario, result).passed is True


def test_routing_fails_on_wrong_type():
    scenario = _scenario({"routes": ["update_todos"]})
    result = RunResult(scenario_id="s", routes=["update_profile"])
    score = score_routing(scenario, result)
    assert score.passed is False
    assert "update_profile" in score.detail


def test_routing_detects_over_saving():
    """Saving when nothing should have been saved is the failure mode the
    'lean toward capturing tasks' prompt makes likely."""
    scenario = _scenario({"routes": []})
    result = RunResult(scenario_id="s", routes=["update_todos"])
    assert score_routing(scenario, result).passed is False


def test_routing_detects_missed_save():
    scenario = _scenario({"routes": ["update_todos"]})
    result = RunResult(scenario_id="s", routes=[])
    assert score_routing(scenario, result).passed is False


def test_routing_not_applicable_without_expectation():
    assert score_routing(_scenario({}), RunResult(scenario_id="s")).passed is None


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #


def _integrity_run(final_todo):
    seed = {
        "todo": [
            {"task": "vet appointment", "status": "not started"},
            {"task": "renew passport", "status": "not started"},
        ]
    }
    scenario = _scenario(
        {
            "changed_records": 1,
            "unchanged": ["renew passport"],
            "field_values": {"vet appointment": {"status": "done"}},
        },
        layer="integrity",
        seed=seed,
    )
    result = RunResult(
        scenario_id="s",
        seed_memory={
            "todo": {
                "seed-todo-0": {"task": "vet appointment", "status": "not started"},
                "seed-todo-1": {"task": "renew passport", "status": "not started"},
            }
        },
        final_memory={"todo": final_todo},
    )
    return scenario, result


def test_integrity_passes_on_clean_patch():
    scenario, result = _integrity_run(
        {
            "seed-todo-0": {"task": "vet appointment", "status": "done"},
            "seed-todo-1": {"task": "renew passport", "status": "not started"},
        }
    )
    assert score_integrity(scenario, result).passed is True


def test_integrity_catches_a_mangled_bystander():
    """The failure JSON-patching exists to prevent: an untouched record quietly
    rewritten while a different one was being updated."""
    scenario, result = _integrity_run(
        {
            "seed-todo-0": {"task": "vet appointment", "status": "done"},
            "seed-todo-1": {"task": "renew passport", "status": "in progress"},
        }
    )
    score = score_integrity(scenario, result)
    assert score.passed is False
    assert "renew passport" in score.detail


def test_integrity_catches_a_dropped_bystander():
    scenario, result = _integrity_run(
        {"seed-todo-0": {"task": "vet appointment", "status": "done"}}
    )
    score = score_integrity(scenario, result)
    assert score.passed is False
    assert "renew passport" in score.detail


def test_integrity_catches_wrong_changed_count():
    scenario, result = _integrity_run(
        {
            "seed-todo-0": {"task": "vet appointment", "status": "done"},
            "seed-todo-1": {"task": "renew passport", "status": "not started"},
            "new-key": {"task": "something invented", "status": "not started"},
        }
    )
    score = score_integrity(scenario, result)
    assert score.passed is False
    assert "changed" in score.detail.lower()


def test_integrity_catches_wrong_field_value():
    scenario, result = _integrity_run(
        {
            "seed-todo-0": {"task": "vet appointment", "status": "in progress"},
            "seed-todo-1": {"task": "renew passport", "status": "not started"},
        }
    )
    score = score_integrity(scenario, result)
    assert score.passed is False
    assert "status" in score.detail


def test_integrity_tolerates_rephrased_task_text():
    """The model may reword a task; locating it must not depend on exact text."""
    scenario, result = _integrity_run(
        {
            "seed-todo-0": {"task": "Vet Appointment for dog", "status": "done"},
            "seed-todo-1": {"task": "renew passport", "status": "not started"},
        }
    )
    assert score_integrity(scenario, result).passed is True


def test_integrity_not_applicable_without_expectation():
    assert score_integrity(_scenario({}), RunResult(scenario_id="s")).passed is None


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def test_retrieval_passes_when_prompt_contains_memory():
    scenario = _scenario({"prompt_contains": ["Kratika"]}, layer="retrieval")
    result = RunResult(scenario_id="s", prompts=["...profile: Kratika..."])
    assert score_retrieval(scenario, result).passed is True


def test_retrieval_fails_when_memory_absent_from_prompt():
    scenario = _scenario({"prompt_contains": ["Kratika"]}, layer="retrieval")
    result = RunResult(scenario_id="s", prompts=["nothing useful here"])
    score = score_retrieval(scenario, result)
    assert score.passed is False
    assert "Kratika" in score.detail


def test_retrieval_catches_cross_user_leakage():
    scenario = _scenario({"prompt_excludes": ["not yours"]}, layer="retrieval")
    result = RunResult(scenario_id="s", prompts=["todo: not yours"])
    assert score_retrieval(scenario, result).passed is False


def test_retrieval_is_case_insensitive():
    scenario = _scenario({"prompt_contains": ["kratika"]}, layer="retrieval")
    result = RunResult(scenario_id="s", prompts=["Profile: Kratika"])
    assert score_retrieval(scenario, result).passed is True


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_score_all_marks_a_failed_run_as_failing():
    scenario = _scenario({"routes": ["update_todos"]})
    result = RunResult(scenario_id="s", error="RuntimeError: boom")
    scores = score_all(scenario, result)
    assert all(s.passed is False for s in scores)
    assert any("boom" in s.detail for s in scores)


def test_score_all_returns_only_applicable_layers():
    scenario = _scenario({"routes": ["update_todos"]})
    result = RunResult(scenario_id="s", routes=["update_todos"])
    scores = score_all(scenario, result)
    assert [s.layer for s in scores] == ["routing"]
