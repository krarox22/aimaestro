import pytest
from langchain_core.messages import AIMessage

from aimaestro import graph as graph_module
from aimaestro.graph import (
    INSTRUCTIONS_KEY,
    INSTRUCTIONS_NS,
    PROFILE_NS,
    TODO_NS,
    build_graph,
)
from aimaestro.schemas import Profile, ToDo
from aimaestro.store import MemoryBackend
from tests.conftest import FakeToolCallingModel, StubExtractor, update_memory_call

CONFIG = {"configurable": {"user_id": "kratika", "thread_id": "t1"}}


@pytest.fixture
def fake_model(monkeypatch):
    def install(responses):
        model = FakeToolCallingModel(responses=responses)
        monkeypatch.setattr(graph_module, "get_model", lambda model_id: model)
        return model

    return install


def test_plain_reply_writes_nothing(db_path, fake_model):
    fake_model([AIMessage(content="Hi, what can I help with?")])
    with MemoryBackend(db_path) as backend:
        app = build_graph(backend)
        result = app.invoke({"messages": [("user", "hello")]}, CONFIG)
        assert backend.store.search((TODO_NS, "kratika")) == []
    assert "help" in result["messages"][-1].content


def test_profile_update_is_persisted(db_path, fake_model, monkeypatch):
    fake_model([update_memory_call("user"), AIMessage(content="Good to know.")])
    monkeypatch.setattr(
        graph_module,
        "get_profile_extractor",
        lambda model_id: StubExtractor(
            [Profile(name="Kratika", interests=["langgraph"])]
        ),
    )
    with MemoryBackend(db_path) as backend:
        app = build_graph(backend)
        app.invoke({"messages": [("user", "I'm Kratika and I like langgraph")]}, CONFIG)
        saved = [i.value for i in backend.store.search((PROFILE_NS, "kratika"))]
    assert saved[0]["name"] == "Kratika"


def test_todo_update_is_persisted(db_path, fake_model, monkeypatch):
    fake_model([update_memory_call("todo"), AIMessage(content="Added it.")])
    monkeypatch.setattr(
        graph_module,
        "get_todo_extractor",
        lambda model_id, listener=None: StubExtractor([ToDo(task="renew passport")]),
    )
    with MemoryBackend(db_path) as backend:
        app = build_graph(backend)
        app.invoke({"messages": [("user", "I need to renew my passport")]}, CONFIG)
        saved = [i.value for i in backend.store.search((TODO_NS, "kratika"))]
    assert saved[0]["task"] == "renew passport"


def test_instructions_update_is_persisted(db_path, fake_model):
    fake_model(
        [
            update_memory_call("instructions"),
            AIMessage(content="Always include a deadline."),
            AIMessage(content="Understood."),
        ]
    )
    with MemoryBackend(db_path) as backend:
        app = build_graph(backend)
        app.invoke({"messages": [("user", "always add deadlines")]}, CONFIG)
        saved = backend.store.get((INSTRUCTIONS_NS, "kratika"), INSTRUCTIONS_KEY)
    assert saved is not None and "deadline" in saved.value["memory"].lower()


def test_memory_is_visible_to_a_later_session(db_path, fake_model, monkeypatch):
    """Write a todo through one graph, read it back through a freshly built one."""
    fake_model([update_memory_call("todo"), AIMessage(content="Added.")])
    monkeypatch.setattr(
        graph_module,
        "get_todo_extractor",
        lambda model_id, listener=None: StubExtractor([ToDo(task="book vet")]),
    )
    with MemoryBackend(db_path) as backend:
        build_graph(backend).invoke({"messages": [("user", "book the vet")]}, CONFIG)

    with MemoryBackend(db_path) as backend:
        saved = [i.value for i in backend.store.search((TODO_NS, "kratika"))]
    assert saved[0]["task"] == "book vet"


def test_stored_memory_reaches_the_prompt(db_path, fake_model):
    """A saved profile and todo must actually be shown to the model next turn.

    Without this, memory could persist correctly and still never influence a
    single reply.
    """
    model = fake_model([AIMessage(content="Welcome back.")])
    with MemoryBackend(db_path) as backend:
        backend.store.put((PROFILE_NS, "kratika"), "p1", {"name": "Kratika"})
        backend.store.put((TODO_NS, "kratika"), "t1", {"task": "renew passport"})
        build_graph(backend).invoke({"messages": [("user", "hi")]}, CONFIG)

    system_prompt = model.received[0][0].content
    assert "Kratika" in system_prompt
    assert "renew passport" in system_prompt


def test_memory_is_scoped_to_the_user(db_path, fake_model):
    """Another user's memories must not leak into this user's prompt."""
    model = fake_model([AIMessage(content="Hello.")])
    with MemoryBackend(db_path) as backend:
        backend.store.put((TODO_NS, "someone-else"), "t1", {"task": "not yours"})
        build_graph(backend).invoke({"messages": [("user", "hi")]}, CONFIG)

    assert "not yours" not in model.received[0][0].content


def test_extraction_failure_does_not_crash_the_chat(db_path, fake_model, monkeypatch):
    fake_model([update_memory_call("todo"), AIMessage(content="Sorry, trouble saving.")])

    class Exploding:
        def invoke(self, payload):
            raise RuntimeError("model refused")

        def with_listeners(self, **kwargs):
            return self

    monkeypatch.setattr(
        graph_module, "get_todo_extractor", lambda model_id, listener=None: Exploding()
    )
    with MemoryBackend(db_path) as backend:
        result = build_graph(backend).invoke(
            {"messages": [("user", "add a task")]}, CONFIG
        )
    assert result["messages"][-1].content  # conversation continued
