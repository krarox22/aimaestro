from aimaestro.store import MemoryBackend


def test_put_and_search(tmp_path):
    db = str(tmp_path / "t.db")
    with MemoryBackend(db) as backend:
        backend.store.put(("profile", "kratika"), "p1", {"name": "Kratika"})
        found = backend.store.search(("profile", "kratika"))
    assert [i.value for i in found] == [{"name": "Kratika"}]


def test_memory_survives_restart(tmp_path):
    """The central guarantee: memory outlives the process that wrote it."""
    db = str(tmp_path / "t.db")

    with MemoryBackend(db) as backend:
        backend.store.put(("todo", "kratika"), "t1", {"task": "renew passport"})

    # Entirely new backend, new connections, same file.
    with MemoryBackend(db) as backend:
        found = backend.store.search(("todo", "kratika"))

    assert [i.value for i in found] == [{"task": "renew passport"}]


def test_checkpointer_and_store_share_a_file(tmp_path):
    db = str(tmp_path / "t.db")
    with MemoryBackend(db) as backend:
        assert backend.store is not None
        assert backend.checkpointer is not None
        backend.store.put(("profile", "u"), "k", {"a": 1})
        assert backend.store.get(("profile", "u"), "k").value == {"a": 1}


def test_creates_parent_directory(tmp_path):
    db = str(tmp_path / "nested" / "dir" / "t.db")
    with MemoryBackend(db) as backend:
        backend.store.put(("profile", "u"), "k", {"a": 1})
    assert (tmp_path / "nested" / "dir" / "t.db").exists()


def test_namespaces_are_isolated_per_user(tmp_path):
    db = str(tmp_path / "t.db")
    with MemoryBackend(db) as backend:
        backend.store.put(("todo", "alice"), "t1", {"task": "alice task"})
        backend.store.put(("todo", "bob"), "t1", {"task": "bob task"})
        alice = [i.value["task"] for i in backend.store.search(("todo", "alice"))]
        bob = [i.value["task"] for i in backend.store.search(("todo", "bob"))]
    assert alice == ["alice task"]
    assert bob == ["bob task"]
