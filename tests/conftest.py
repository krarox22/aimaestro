"""Test doubles that let the whole graph run without an API key."""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aimaestro import graph as graph_module


class RealModelForbidden(BaseException):
    """Raised when a test tries to reach a real provider.

    Derives from BaseException, not Exception, on purpose: `evals.runner`
    deliberately catches every Exception so a broken run becomes data rather
    than a crash. That is right for evals and wrong here — it would swallow this
    guard and let a money-spending test report green. BaseException sails
    straight through, the same way pytest's own control-flow exceptions do.
    """


@pytest.fixture(autouse=True)
def forbid_real_models(monkeypatch):
    """Make it structurally impossible for the test suite to call a provider.

    The test suite and the eval harness are different tools: tests assert
    deterministic behavior against fakes, evals score a real model and cost
    money. Both are importable from here, so without this guard a single
    careless test — one that calls `evals.cli.main` with a key present — would
    quietly start billing on every `pytest` run.

    Real-model runs belong in `python -m evals`, and nowhere else.
    """
    _clear_model_cache()

    def _forbidden(model_id, *args, **kwargs):
        raise RealModelForbidden(
            f"A test tried to build a real model ({model_id!r}). "
            "Tests run against fakes; real-model runs belong in "
            "`python -m evals`."
        )

    monkeypatch.setattr(graph_module, "init_chat_model", _forbidden)
    yield
    # Teardown runs before monkeypatch restores, so get_model may still be a
    # test's plain stub with no cache to clear.
    _clear_model_cache()


def _clear_model_cache() -> None:
    cache_clear = getattr(graph_module.get_model, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


class FakeToolCallingModel(BaseChatModel):
    """Replays a scripted list of AIMessages, one per invocation.

    The final message repeats once the script runs out, so a graph that loops
    back to the conversational node terminates instead of hanging.
    """

    responses: list[AIMessage] = []
    calls: int = 0
    #: Every message list this model was handed, so tests can assert on prompts.
    received: list[list] = []

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received.append(list(messages))
        index = min(self.calls, len(self.responses) - 1)
        message = self.responses[index]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolCallingModel":
        return self


class StubExtractor:
    """Stands in for a trustcall extractor with a canned result."""

    def __init__(self, responses: list[Any], metadata: list[dict] | None = None):
        self._responses = responses
        self._metadata = metadata or [{} for _ in responses]
        self.invocations: list[dict] = []

    def invoke(self, payload: dict) -> dict:
        self.invocations.append(payload)
        return {"responses": self._responses, "response_metadata": self._metadata}

    def with_listeners(self, **kwargs: Any) -> "StubExtractor":
        return self


def update_memory_call(update_type: str, call_id: str = "call-1") -> AIMessage:
    """An AIMessage shaped like the model asking to write memory."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "UpdateMemory", "args": {"update_type": update_type}, "id": call_id}
        ],
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "aimaestro.db")
