"""Test doubles that let the whole graph run without an API key."""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


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
