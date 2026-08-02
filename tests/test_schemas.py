from typing import get_args, get_type_hints

from aimaestro.schemas import Profile, ToDo, UpdateMemory


def test_profile_all_fields_optional():
    p = Profile()
    assert p.name is None
    assert p.connections == []
    assert p.interests == []


def test_todo_requires_only_task():
    t = ToDo(task="renew passport")
    assert t.task == "renew passport"
    assert t.status == "not started"
    assert t.solutions == []
    assert t.deadline is None


def test_todo_solutions_accepts_empty_list():
    """Regression: the original schema declared min_items=1 alongside an empty
    default, which are contradictory."""
    t = ToDo(task="x", solutions=[])
    assert t.solutions == []


def test_update_memory_literal_values():
    hints = get_type_hints(UpdateMemory)
    assert set(get_args(hints["update_type"])) == {"user", "todo", "instructions"}
