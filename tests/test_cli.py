from aimaestro.cli import format_memory
from aimaestro.store import MemoryBackend


def test_format_memory_empty(db_path):
    with MemoryBackend(db_path) as backend:
        text = format_memory(backend.store, "kratika")
    assert "nothing" in text.lower()


def test_format_memory_shows_all_three(db_path):
    with MemoryBackend(db_path) as backend:
        backend.store.put(("profile", "kratika"), "p", {"name": "Kratika"})
        backend.store.put(("todo", "kratika"), "t", {"task": "renew passport"})
        backend.store.put(
            ("instructions", "kratika"),
            "user_instructions",
            {"memory": "add deadlines"},
        )
        text = format_memory(backend.store, "kratika")
    assert "Kratika" in text
    assert "renew passport" in text
    assert "add deadlines" in text


def test_format_memory_omits_empty_profile_fields(db_path):
    with MemoryBackend(db_path) as backend:
        backend.store.put(
            ("profile", "kratika"),
            "p",
            {"name": "Kratika", "job": None, "interests": []},
        )
        text = format_memory(backend.store, "kratika")
    assert "job" not in text
    assert "interests" not in text


def test_format_memory_is_scoped_to_user(db_path):
    with MemoryBackend(db_path) as backend:
        backend.store.put(("todo", "someone-else"), "t", {"task": "not yours"})
        text = format_memory(backend.store, "kratika")
    assert "not yours" not in text
